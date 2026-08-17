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
            stop_threshold=500):
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

    while sim.step < steps_target:
        due = (wounding_active and wound_start_step is not None
               and sim.step >= wound_start_step
               and (last_wound_step is None
                    or sim.step - last_wound_step >= wound_freq_steps))
        if due:
            sim.inflict_wound(False)
            wounds_inflicted += 1
            last_wound_step = sim.step
            if scenario == 'bounded':
                clone_now = sim.largest_clone()
                if clone_now > stop_threshold:
                    clone_established_at_h = sim.step * K['dtMin'] / 60
                    wounding_active = False  # permanent stop for this run
        sim.advance()

    final_clone = sim.largest_clone()

    return dict(
        run_id=run_id, scenario=scenario, seed=seed,
        tau_d=params.get('lifeD'), cyc_h=params.get('cycH', 15.0),
        Twake=params.get('Twake', 1.08), sigma=params.get('sigma', .85),
        pRevert=params.get('pRevert', 1 / 1000),
        wound_start_h=wound_start_h, wound_freq_h=wound_freq_h,
        wound_size=wound_size, wounds_inflicted=wounds_inflicted,
        stop_threshold=(stop_threshold if scenario == 'bounded' else None),
        clone_established_at_h=clone_established_at_h,
        target_hours=target_hours, elapsed_h=sim.step * K['dtMin'] / 60,
        final_clone_size=final_clone,
        rev_total=sim.rev_total, rev_lost=sim.rev_lost,
        shed_total=sim.shed_total, div_total=sim.div_total,
    )


if __name__ == '__main__':
    # FIRST TIMED TEST — 1 seed per scenario, on purpose. The goal right now
    # is to measure real wall-clock time on YOUR hardware, not to produce a
    # usable N for statistics yet. Once you see the actual timings, we scale
    # N up (or trim target_hours) based on real numbers, not guesses.
    TARGET_HOURS = 20000
    WOUND_START_H = 1000
    WOUND_SIZE = 200
    FREQ_CHOICES = [12, 24, 48]
    STOP_THRESHOLD = 500
    SCENARIOS = ['none', 'chronic', 'bounded']
    N_SEEDS = 1

    OUT_CSV = 'runs_test_2026-08-05.csv'   # rename per day/batch as you like

    if not HAVE_NUMBA:
        print("*** WARNING: numba is NOT installed — the engine will run in "
              "pure-Python fallback mode, which is far SLOWER than the old "
              "NumPy version, let alone the compiled one. Run "
              "`!pip install numba` first, or switch the import at the top "
              "of this file back to epithelium_headless. ***")

    N_WORKERS = os.cpu_count() or 1
    print(f"Using {N_WORKERS} worker processes (os.cpu_count()).")

    jobs = []
    run_id = 0
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            run_id += 1
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

    t0 = time.time()
    writer = None
    f = open(OUT_CSV, 'w', newline='')
    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(run_one, **job): job for job in jobs}
            for fut in as_completed(futures):
                row = fut.result()
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f"run_id={row['run_id']} scenario={row['scenario']} "
                      f"seed={row['seed']} -> final_clone={row['final_clone_size']} "
                      f"established_at={row['clone_established_at_h']} "
                      f"({time.time()-t0:.1f}s elapsed)")
    finally:
        f.close()

    print(f"\nwrote results to {OUT_CSV} in {time.time()-t0:.1f}s total")
