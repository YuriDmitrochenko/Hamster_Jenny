# Monolayer Epithelium — Headless Simulation Engine (Draft)

> **Status: draft / sketch.** This code reproduces the deterministic macro-behaviour of the
> reference model (see Validation below) but has not yet been run at full scale
> (target: N = 100–500 seeds per parameter combination, 20,000 h horizon). Treat
> results from this version as a workflow check, not a scientific conclusion.

## What this is

A headless (no-graphics) Python port of an interactive HTML/JS simulation stand
modelling a single-layer epithelial monolayer as a territory-based cellular
automaton: cells hold 1–4 grid nodes, shed on a stochastic lifespan, spread into
freed nodes, redistribute area under tension, and a reserve (quiescent)
subpopulation wakes and divides once local tension crosses a threshold.

A background stochastic process lets a division daughter revert to an
undifferentiated, non-shedding lineage; whether that reverted lineage survives,
forms its own niche, or dies is decided by its local neighbourhood, not by the
reversion event itself.

The purpose of this port is to run the model many times (varying seed, τ,
wound/no-wound, etc.) and export tabular results for statistical analysis —
something the original visual stand (kept separately for illustration figures)
isn't built for.

This is companion code to a pilot article studying two questions: **system
stability** (deterministic reserve/density balance) and **the appearance and
regression of a reverted clone following injury** (stochastic clone
establishment).

## Contents

| File | What it does |
|---|---|
| `epithelium_headless.py` | The simulation engine: cell-state arrays, one `advance()` step, wound infliction, `count()` / `burden()` readouts. No plotting, no UI. |
| `pilot_batch_draft.py` | A small, fast **draft** batch runner built on the engine above: tracks reverted-clone "episodes" over time and writes two CSV tables. Sized deliberately small (few seeds, short horizon) to sanity-check the workflow before committing to a full batch. |

## Requirements

```
numpy
pandas
```
Nothing else — both files run as plain Python scripts, no notebook-specific
setup required (tested standalone and in DataCamp DataLab).

## Running it

```bash
python pilot_batch_draft.py
```

Runs a small grid (currently `τ = 6 d`, wound vs. no-wound, a couple of seeds
each, short horizon) and writes two files to the working directory:
`runs_sketch.csv` and `episodes_sketch.csv`. Edit the constants at the bottom
of `pilot_batch_draft.py` (`TAUS`, `WOUNDS`, `N_SEEDS`, `TARGET_HOURS`, ...) to
change the grid — there is no CLI yet.

`epithelium_headless.py` can also be driven directly for a single run:

```python
from epithelium_headless import Sim

sim = Sim(seed=1, params=dict(lifeD=6.0))
for _ in range(1000):
    sim.advance()
print(sim.count())
```

## Output schema

Two tables, joined by `run_id` (long/tidy format — a run can have zero, one,
or many clone episodes, so a wide one-column-per-episode layout was
deliberately avoided):

**`runs_sketch.csv`** — one row per simulation run
| column | meaning |
|---|---|
| `run_id` | join key |
| `seed` | RNG seed for this run |
| `tau_d`, `cyc_h`, `Twake`, `sigma`, `pRevert` | model parameters used |
| `has_wound`, `wound_at_h`, `defect` | wound protocol for this run, if any |
| `target_hours`, `confirm_h` | run configuration |
| `elapsed_h` | simulated hours actually completed |
| `n_episodes` | number of reverted-clone episodes detected |
| `final_burden` | reverted-lineage cell count at the end of the run |
| `final_status` | `extinct` or `active_at_cutoff` |
| `rev_total`, `rev_lost`, `shed_total`, `div_total` | bookkeeping counters |

**`episodes_sketch.csv`** — one row per reverted-clone episode
| column | meaning |
|---|---|
| `run_id` | join key back to `runs_sketch.csv` |
| `episode_num` | 1, 2, 3, ... within that run |
| `start_h` | when burden first rose above 0 (debounced) |
| `peak_burden`, `peak_h` | largest burden reached in this episode, and when |
| `end_h` | when burden returned to 0 (debounced), or the run's cutoff time |
| `censored` | `True` if the episode was still open when the run ended — i.e. we stopped watching, not that the clone necessarily resolved at that exact hour |

**Burden**, here, is the total count of live reverted-lineage cells anywhere in
the tissue at a given time — not just the largest connected patch. This
matches the clinical framing behind the model (mosaic dysplasia / total area
of involvement), rather than treating a new clone at a different location as
a separate thing from a clone that regrew after regressing.

## Validation

After fixing a step-ordering bug (a newly-woken reserve cell was being
re-processed as an already-dividing cell within the same step and reverting
immediately), a plain run at τ = 6 d converges to density ≈ 53% and reserve
mobilization ≈ 97%, matching the reference table from the original visual
stand and the draft article (52% / 98%). No further validation against the
visual stand has been done yet.

## Known simplifications (read before trusting results)

- **Synchronous approximation.** The original stand updates "spreading"
  (claim) and "shift" (tension redistribution) one grid node at a time, in
  random order, mutating state between picks. This port instead computes a
  batch of candidate updates against a single pre-step snapshot and applies
  them together (with de-duplication to keep territory sizes within
  `[0, aMax]`). This preserves the direction and rough magnitude of both
  processes but is **not** bit-identical to the visual stand's dynamics.
  Recommended before trusting a large batch: run a short (e.g. 500 h) side
  by side comparison against the visual stand.
- **Episode debounce is a first draft.** `confirm_h` (default 24 h) is meant
  to stop single-step burden flicker from being counted as a full episode
  start/end, but a flip that reverses before the debounce window elapses
  currently just resets the counter rather than remembering partial
  progress. Not yet stress-tested against real batch data.
- **No checkpointing yet.** `pilot_batch_draft.py` holds all rows in memory
  and writes CSVs only at the very end — fine for a small sketch, not
  suitable for a long batch that may be split across many short sessions.
  This is the next thing to add before a full-scale run.

## Performance

No JIT (Numba/Cython) in this version. A single run to 20,000 h takes roughly
50 minutes on one CPU core; plan batch size and session count accordingly.

## Authors

Yuri Dmitrochenko, MD · Sarah-Lana Demi

Developed with code assistance from Claude (Anthropic). Companion code to
the manuscript-in-progress on monolayer epithelial dynamics and
reverted-clone establishment.

## Citing this code

Once archived, add the Zenodo DOI badge and citation here, e.g.:

```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

A `CITATION.cff` file is recommended alongside this README so GitHub renders
a "Cite this repository" button automatically.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
— see `LICENSE`. (Per Creative Commons' own guidance not to apply CC
licenses to software.) The manuscript is licensed and archived separately,
not as part of this repository.
