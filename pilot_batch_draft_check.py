"""
Batch runner — Stage 1/2 initial timed test, full 20000h horizon.

Metric: ONLY the size (cell count) of the largest CONNECTED reverted clone
(Sim.largest_clone(), the BFS-based patch size — not total tissue-wide
burden). Measured only at the very end of the run, plus at each scheduled
wound-decision point for the 'bounded' scenario (needed to detect the
500-cell stop trigger). Nothing about persistence duration is tracked; a
clone fluctuating along the way doesn't matter, only where it ends up.

No episode log, no episodes.csv — dropped per the Aug 4 decision, since
clone identity/continuity/location doesn't matter clinically.

Three wound scenarios, chosen per run_id (see SCENARIOS below):
  - 'none'    — no wound at all (Stage 1: can a large clone arise by
                chance alone?)
  - 'chronic' — wound repeats every wound_freq_h, starting at
                wound_start_h, for the entire run — never stops, regardless
                of clone size.
  - 'bounded' — same repeating wound, but stops permanently the first time
                the largest connected clone exceeds stop_threshold cells;
                the run keeps simulating afterwards with no further wounds,
                so you can see what happens once the injury stops.

wound_freq_h is drawn once per run from {12, 24, 48} h (fixed for that run,
not re-drawn per wound event) — seeded from the run's own seed, so it's
reproducible.

Output: ONE CSV, rows appended as each run finishes (works fine under
multiprocessing — only the main process writes, workers just return a row).
Name the file per test batch/date, e.g. runs_test_2026-08-05.csv; run_id +
scenario + seed inside the rows is enough to tell runs apart without
needing one file per run.

Performance note (read before running): a 20000h run with NO wound took
about 50 minutes on one CPU core in Claude's sandbox. Repeated wounding
throughout the whole run is new territory — likely slower than that
per-run, by an unknown amount until measured on your actual hardware.
Runs are parallelized across processes (one scenario per core, for this
small first test), so total wall time is roughly the time of the SLOWEST
of the three, not the sum — but "finishes within an hour" is not
guaranteed and needs to be checked empirically.
"""
import os
import csv
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
# Engine switch. epithelium_numba is the fast one (needs `pip install numba`;
# falls back to slow-but-correct pure Python if numba is missing). Change this
# line to `from epithelium_headless import Sim, K` to use the older NumPy port.
from epithelium_numba import Sim, K, HAVE_NUMBA


def run_one(run_id, params, seed, target_hours, scenario,
            wound_start_h=None, wound_freq_h=None, wound_size=None,
            stop_threshold=500, check_every_h=24.0):
    p = dict(params)
    if wound_size is not None:
        p['defect'] = wound_size
    sim = Sim(seed, p)
    steps_target = int(round(target_hours * 60 / K['dtMin']))

    wound_start_step = (int(round(wound_start_h * 60 / K['dtMin']))
                         if wound_start_h is not None else None)
    wound_freq_steps = (int(round(wound_freq_h * 60 / K['dtMin']))
                         if wound_freq_h is not None else None)

    wounding_active = scenario in ('chronic', 'bounded')
    wounds_inflicted = 0
    clone_established_at_h = None
    last_wound_step = None

    # The threshold is checked on a FIXED cadence in every scenario, so the
    # crossing time means the same thing across all three and stays comparable
    # (previously it was only checked at wound events, so 'none' could never
    # record one and 'chronic' inherited the wound frequency as its sampling
    # rate). Only 'bounded' acts on the crossing by stopping the wounding;
    # elsewhere it is recorded and nothing else happens.
    check_every_steps = max(1, int(round(check_every_h * 60 / K['dtMin'])))

    while sim.step < steps_target:
        due = (wounding_active and wound_start_step is not None
               and sim.step >= wound_start_step
               and (last_wound_step is None
                    or sim.step - last_wound_step >= wound_freq_steps))
        if due:
            sim.inflict_wound(False)
            wounds_inflicted += 1
            last_wound_step = sim.step
        sim.advance()

        if clone_established_at_h is None and sim.step % check_every_steps == 0:
            if sim.largest_clone() > stop_threshold:
                clone_established_at_h = sim.step * K['dtMin'] / 60
                if scenario == 'bounded':
                    wounding_active = False   # permanent stop for this run

    final_clone = sim.largest_clone()
    final_count = sim.count()

    return dict(
        run_id=run_id, scenario=scenario, seed=seed,
        tau_d=params.get('lifeD'), cyc_h=params.get('cycH', 15.0),
        Twake=params.get('Twake', 1.08), sigma=params.get('sigma', .85),
        pRevert=params.get('pRevert', 1 / 1000),
        wound_start_h=wound_start_h, wound_freq_h=wound_freq_h,
        wound_size=wound_size, wounds_inflicted=wounds_inflicted,
        stop_threshold=stop_threshold, check_every_h=check_every_h,
        wounding_stops_on_threshold=(scenario == 'bounded'),
        clone_established_at_h=clone_established_at_h,
        target_hours=target_hours, elapsed_h=sim.step * K['dtMin'] / 60,
        final_clone_size=final_clone,
        # clone size alone can't be read without knowing how big the sheet
        # still is — under chronic wounding the cell population is far below
        # the 6004-node maximum, so 2500 cells is a much larger share than it
        # looks. final_clone_frac is the share of surviving cells.
        final_cells_total=final_count['cells'],
        final_density=final_count['dens'],
        final_clone_frac=(final_clone / final_count['cells']
                          if final_count['cells'] else 0.0),
        final_burden=sim.burden(),
        rev_total=sim.rev_total, rev_lost=sim.rev_lost,
        shed_total=sim.shed_total, div_total=sim.div_total,
    )


if __name__ == '__main__':
    # REAL SERIES — 100 seeds per scenario (300 runs total).
    # At the measured 0.42 ms/step this is ~2.8 min per run single-core,
    # i.e. ~14 core-hours; divided across os.cpu_count() workers.
    # Seeds are 0..N_SEEDS-1 and identical across scenarios on purpose, so
    # each scenario sees the same set of initial tissues — a paired design,
    # which removes some between-scenario variance.
    TARGET_HOURS = 20000
    WOUND_START_H = 1000
    WOUND_SIZE = 200
    FREQ_CHOICES = [12, 24, 48]
    STOP_THRESHOLD = 500
    SCENARIOS = ['none', 'chronic', 'bounded']
    N_SEEDS = 100

    OUT_CSV = 'runs_series_2026-08-17.csv'   # keep the SAME name across sessions
                                              # — that's what makes resuming work

    # Session budget: the script stops launching new runs once this much time
    # has passed, so you can do the series in short sittings. Set to None for
    # "run everything in one go". Runs already finished are in the CSV and are
    # skipped on the next launch, so just re-run the same command tomorrow.
    SESSION_MINUTES = 30

    if not HAVE_NUMBA:
        print("*** WARNING: numba is NOT installed — the engine will run in "
              "pure-Python fallback mode, which is far SLOWER than the old "
              "NumPy version, let alone the compiled one. Run "
              "`!pip install numba` first, or switch the import at the top "
              "of this file back to epithelium_headless. ***")

    N_WORKERS = os.cpu_count() or 1
    print(f"Using {N_WORKERS} worker processes (os.cpu_count()).")

    # ---- what's already done? (resume) ----
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline='') as fh:
            for r in csv.DictReader(fh):
                done.add((r['scenario'], int(r['seed'])))
        print(f"found {len(done)} completed runs in {OUT_CSV} — skipping those")

    # run_id is derived from (scenario, seed) rather than from enumeration
    # order, so it stays stable if a later session runs only a subset of
    # scenarios (e.g. a no-wound-only extension). Without this, a run with
    # SCENARIOS=['none'] would restart numbering at 1 and collide with
    # existing rows. The real key is still (scenario, seed) — use that when
    # joining in pandas, not run_id.
    ALL_SCENARIOS = ['none', 'chronic', 'bounded']

    jobs = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            run_id = ALL_SCENARIOS.index(scenario) * 100000 + seed
            if (scenario, seed) in done:
                continue
            if scenario == 'none':
                freq = None
            else:
                freq = random.Random(seed).choice(FREQ_CHOICES)
            jobs.append(dict(
                run_id=run_id, params=dict(lifeD=6.0), seed=seed,
                target_hours=TARGET_HOURS, scenario=scenario,
                wound_start_h=(WOUND_START_H if scenario != 'none' else None),
                wound_freq_h=freq,
                wound_size=(WOUND_SIZE if scenario != 'none' else None),
                stop_threshold=STOP_THRESHOLD,
            ))

    total = len(SCENARIOS) * N_SEEDS
    print(f"{len(jobs)} runs left of {total}")
    if not jobs:
        print("nothing to do — the series is complete.")
        raise SystemExit

    t0 = time.time()
    new_file = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    f = open(OUT_CSV, 'a', newline='')
    writer = None
    completed = 0
    stopped_early = False
    try:
        # Jobs go out in waves of N_WORKERS so the time budget can be checked
        # between waves. A run in flight is never killed — that would waste the
        # minutes already spent on it — so the session overruns the budget by
        # up to one run's duration (~3 min).
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for start in range(0, len(jobs), N_WORKERS):
                if (SESSION_MINUTES is not None
                        and (time.time() - t0) / 60 >= SESSION_MINUTES):
                    stopped_early = True
                    break
                wave = jobs[start:start + N_WORKERS]
                futures = [ex.submit(run_one, **job) for job in wave]
                for fut in as_completed(futures):
                    row = fut.result()
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if new_file:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    completed += 1
                    print(f"[{completed}/{len(jobs)}] {row['scenario']} "
                          f"seed={row['seed']} -> clone={row['final_clone_size']} "
                          f"frac={row['final_clone_frac']:.3f} "
                          f"est_at={row['clone_established_at_h']} "
                          f"({(time.time()-t0)/60:.1f} min)")
    finally:
        f.close()

    left = len(jobs) - completed
    print(f"\n{completed} runs done this session in "
          f"{(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    if stopped_early or left:
        print(f"{left} runs still to go — just run this script again "
              f"(same OUT_CSV) to continue where it stopped.")
    else:
        print("series complete.")
