"""
Draft (sketch) batch runner — small, fast, exploratory.

Purpose: NOT a scientific result. This produces a tiny, quick set of runs so
Sara can load the CSVs in pandas, try merging runs.csv <-> episodes.csv,
and see what's awkward about the schema BEFORE we commit to a real batch
(N=100-500 seeds, full 20000h horizon, which takes real wall-clock time).

Episode definition (burden-based, per the agreed rules):
  - burden(t) = count of all live reverted-lineage cells anywhere in the
    tissue (mosaic-aware — not just the largest connected patch).
  - an episode STARTS when burden crosses 0 -> >0 and stays that way for
    at least `confirm_h` hours (debounce, to ignore single-step noise)
  - an episode ENDS when burden crosses >0 -> 0 and stays that way for at
    least `confirm_h` hours
  - if the run reaches its time limit while burden > 0, the open episode is
    marked censored=True (we simply stopped watching, the hamster didn't
    necessarily "recover" at that exact moment)

This confirm_h debounce is a first draft, not a settled decision — the
window in which a flip-then-flip-back inside the confirm period is currently
just... allowed to re-arm from scratch (see comments below). Sara: this is
exactly the kind of thing to poke at and tell us if it misbehaves.
"""
import time
import numpy as np
import pandas as pd
from epithelium_headless import Sim, K, SHED


def run_one_with_episodes(run_id, params, seed, target_hours,
                           wound_at_h=None, wound_defect=None,
                           confirm_h=24.0):
    p = dict(params)
    if wound_defect is not None:
        p['defect'] = wound_defect
    sim = Sim(seed, p)
    steps_target = int(round(target_hours * 60 / K['dtMin']))
    wound_step = int(round(wound_at_h * 60 / K['dtMin'])) if wound_at_h is not None else None
    confirm_steps = max(1, int(round(confirm_h * 60 / K['dtMin'])))

    episodes = []
    b0 = sim.burden()
    above = b0 > 0
    since_change = 0
    cur_ep = dict(start_h=0.0, peak_burden=b0, peak_h=0.0) if above else None

    while sim.step < steps_target:
        if wound_step is not None and sim.step == wound_step:
            sim.inflict_wound(False)
        sim.advance()
        elapsed_h = sim.step * K['dtMin'] / 60
        b = sim.burden()
        now_above = b > 0

        if now_above != above:
            since_change += 1
            if since_change >= confirm_steps:
                if now_above:
                    # confirmed START — back-date to roughly when burden first
                    # rose (elapsed_h - confirm_h), not to the confirmation
                    # moment itself
                    cur_ep = dict(start_h=max(0.0, elapsed_h - confirm_h),
                                  peak_burden=b, peak_h=elapsed_h)
                else:
                    cur_ep['end_h'] = max(cur_ep['start_h'], elapsed_h - confirm_h)
                    cur_ep['censored'] = False
                    episodes.append(cur_ep)
                    cur_ep = None
                above = now_above
                since_change = 0
        else:
            # NOTE (draft behaviour, flag for review): a flip that reverses
            # before reaching confirm_steps resets the debounce counter to 0
            # rather than remembering partial progress. Good enough for a
            # sketch; revisit if real runs show a lot of borderline flicker.
            since_change = 0

        if above and cur_ep is not None and b > cur_ep['peak_burden']:
            cur_ep['peak_burden'] = b
            cur_ep['peak_h'] = elapsed_h

    if cur_ep is not None:
        cur_ep['end_h'] = sim.step * K['dtMin'] / 60
        cur_ep['censored'] = True
        episodes.append(cur_ep)

    final_burden = sim.burden()
    final_status = 'active_at_cutoff' if final_burden > 0 else 'extinct'

    run_row = dict(
        run_id=run_id, seed=seed,
        tau_d=params.get('lifeD'), cyc_h=params.get('cycH', 15.0),
        Twake=params.get('Twake', 1.08), sigma=params.get('sigma', .85),
        pRevert=params.get('pRevert', 1 / 1000),
        has_wound=wound_at_h is not None,
        wound_at_h=wound_at_h,
        defect=(wound_defect if wound_at_h is not None else None),
        target_hours=target_hours, confirm_h=confirm_h,
        elapsed_h=sim.step * K['dtMin'] / 60,
        n_episodes=len(episodes),
        final_burden=final_burden, final_status=final_status,
        rev_total=sim.rev_total, rev_lost=sim.rev_lost,
        shed_total=sim.shed_total, div_total=sim.div_total,
    )

    ep_rows = []
    for i, ep in enumerate(episodes, start=1):
        ep_rows.append(dict(
            run_id=run_id, episode_num=i,
            start_h=ep['start_h'], peak_burden=ep['peak_burden'],
            peak_h=ep['peak_h'], end_h=ep['end_h'], censored=ep['censored'],
        ))
    return run_row, ep_rows


if __name__ == '__main__':
    # SKETCH SETTINGS — deliberately small & short, just to inspect the CSV
    # shape and workflow. Not sized for any real conclusion.
    TAUS = [6.0]
    WOUNDS = [True, False]
    N_SEEDS = 2
    TARGET_HOURS = 200     # short on purpose: full pilot run is ~50 min/run
                            # at 20000h; this sketch needs to finish in
                            # minutes so Sara can iterate on it today.
    WOUND_AT_H = 80
    DEFECT = 600
    CONFIRM_H = 24

    t0 = time.time()
    run_rows, ep_rows = [], []
    run_id = 0
    for tau in TAUS:
        for has_wound in WOUNDS:
            for seed in range(N_SEEDS):
                run_id += 1
                rr, er = run_one_with_episodes(
                    run_id, dict(lifeD=tau), seed, TARGET_HOURS,
                    wound_at_h=WOUND_AT_H if has_wound else None,
                    wound_defect=DEFECT, confirm_h=CONFIRM_H,
                )
                run_rows.append(rr)
                ep_rows.extend(er)
                print(f"run_id={run_id} tau={tau} wound={has_wound} seed={seed} "
                      f"-> episodes={rr['n_episodes']} final={rr['final_status']} "
                      f"({time.time()-t0:.1f}s elapsed)")

    runs_df = pd.DataFrame(run_rows)
    episodes_df = pd.DataFrame(ep_rows)
    runs_df.to_csv('/mnt/user-data/outputs/runs_sketch.csv', index=False)
    episodes_df.to_csv('/mnt/user-data/outputs/episodes_sketch.csv', index=False)
    print(f"\nwrote {len(runs_df)} rows to runs_sketch.csv")
    print(f"wrote {len(episodes_df)} rows to episodes_sketch.csv")
    print(f"total time: {time.time()-t0:.1f}s")
