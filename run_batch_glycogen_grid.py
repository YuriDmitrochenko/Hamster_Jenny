"""
Combined engine + batch runner — GLYCOGEN GRID under wound, oxygen filter
OFF (oxyRev=1.0): revStore in [0.20, 0.25, 0.30] x scenario in
['bounded', 'chronic'], threshold=500. Extends the revStore=0.33/oxyRev=1.0
baseline (runs_series_baseline_033_2026-09.csv, which already covers all
three scenarios at 0.33) down to lower glycogen values — 'none' is
deliberately NOT repeated here since the 0.33 baseline already showed
100% establishment at 'none' with the filter off; a lower glycogen floor
without any wound is not an interesting question. This series asks
whether a lower glycogen floor changes anything once there IS a wound.

This is epithelium__numba.py (the numba engine) and the batch runner
(pilot_batch_draft_1708.py) merged into ONE file, with the oxyRev
(oxygen filter) mechanism added to the engine:

  - revStore default fixed at .33 (was .2 in the old engine default —
    that stale default is now overridden explicitly here so nothing
    silently reverts to the wrong glycogen value)
  - oxyRev: probability that a newly-created reverted-lineage daughter
    survives the reversion/division event. Rolled once per new reverted
    cell, at the moment it's created (mirrors the JS bench's birth-filter
    mechanism exactly, same RNG order). Fixed at 1.0 for this whole
    series (filter off) — that's the point of this run.
  - new counter rev_oxy_death: how many would-be reverted cells died at
    birth because they failed the O2 roll (will be 0 throughout this
    series, since oxyRev=1.0 means the roll always succeeds — kept in
    the output for schema consistency with the other series).

Wound protocol (bounded / chronic) matches every other series in this
file for comparability: wound_start_h=1000, wound_size=200, wound_freq_h
drawn per-run from {12, 24, 48}h. 'bounded' stops wounding once the clone
first crosses stop_threshold; 'chronic' keeps wounding to the end of the
horizon regardless of clone size. 'none' never wounds at all.

Output goes to a NEW csv (see OUT_CSV below) — deliberately NOT the same
file as any earlier series: the old 300-run series
(runs_series_2026-08-17.csv) used FULL glycogen (revStore=1.0, no oxyRev
column at all); the no-wound oxygrid series
(runs_series_oxygrid_2026-08-29.csv) varies oxyRev and only covers
scenario='none'; the wound-grid series
(runs_series_oxygrid_wound_2026-09.csv) varies oxyRev and only covers
'bounded'/'chronic'. This series is the odd one out: oxyRev is FIXED at
1.0, and all three scenarios are covered, specifically to serve as the
single correct baseline the other two series should be compared against.

BEFORE running the full series: run the short validation at the bottom
of this file (VALIDATE = True) — 1 seed, scenario 'bounded', short horizon,
revStore=0.33 / oxyRev=0.4 — and send rev_total / rev_oxy_death /
final_density / wounds_inflicted / clone_established_at_h back for
cross-check against the JS bench on the same seed. Only flip to the full
series (VALIDATE = False) after that checks out. (The validation run
itself still uses oxyRev=0.4, not 1.0 — it's just a generic engine
sanity check, unrelated to which series you're about to run.)

    pip install numba      # in DataLab, if not already installed
"""
import os
import csv
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in so this file runs (slowly) without numba."""
        def wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrap


# ============================================================
# ENGINE (epithelium__numba.py, with oxyRev added)
# ============================================================

W, H = 76, 79
N = W * H
CAP = N + 8

RESV, DIFF, DIV, SHED = 1, 2, 3, 4

K = dict(P0=.20, Dg=.15, Km=.40, gamma=1.6, dtMin=3, commit=.40, arrTh=.75,
         shedT=8, kClaim=.5, resH=25, beta=12)

# counters[] indices — C_REVOXYDEATH added at the end (was 10 counters, now 11)
C_STEP, C_SHED, C_STARVE, C_DIV, C_REVTOT, C_REVLOST, C_REVEST, C_NEXTID, \
    C_NFREE, C_RNG, C_REVOXYDEATH = range(11)


def build_neighbors():
    NBF = -np.ones((N, 6), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if r & 1:
                T = [(1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)]
            else:
                T = [(1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
            for d, (dx, dy) in enumerate(T):
                nc, nr = c + dx, r + dy
                if 0 <= nc < W and 0 <= nr < H:
                    NBF[i, d] = nr * W + nc
    return NBF


NBF = build_neighbors()


@njit(cache=True)
def _rnd(counters):
    """The JS stand's LCG, so random sequences line up between the two."""
    s = (counters[C_RNG] * 1664525 + 1013904223) & 0xFFFFFFFF
    counters[C_RNG] = s
    return s / 4294967296.0


@njit(cache=True)
def _new_cell(calive, carea, cprog, cshed, split_to, free_ids, counters):
    if counters[C_NFREE] > 0:
        counters[C_NFREE] -= 1
        cid = free_ids[counters[C_NFREE]]
    else:
        cid = counters[C_NEXTID]
        counters[C_NEXTID] += 1
    calive[cid] = 1
    carea[cid] = 0
    cprog[cid] = 0.0
    cshed[cid] = 0
    split_to[cid] = -1
    return cid


@njit(cache=True)
def _kill_cell(cid, calive, cst, clin, free_ids, counters):
    calive[cid] = 0
    cst[cid] = 0
    clin[cid] = 0
    free_ids[counters[C_NFREE]] = cid
    counters[C_NFREE] += 1


@njit(cache=True)
def advance_step(owner, cst, carea, cgly, cprog, cage, clife, cshed, calive,
                 clin, ctol, ctens, g, gn, nbf, free_ids, counters,
                 tsum, tcnt, cup, n_same, n_other, n_free_a, split_to,
                 assigned, dying, dy_flag,
                 aMax, Twake, pResv, pRevert, revStore, oxyRev,
                 c0, Vn, Tcyc, lifeH, lifeSp, resvLifeH, sMax0, tol_steps):
    """One simulation step. Mirrors the JS stand's advance() order exactly."""
    beta = 12.0
    Km = 0.40
    Dg = 0.15
    P0 = 0.20
    dt_h = 3.0 / 60.0
    shedT = 8
    kClaim = 0.5
    commit = 0.40
    arrTh = 0.75

    counters[C_STEP] += 1
    next_id = counters[C_NEXTID]

    # ---- 1. glucose diffusion + consumption ----
    for i in range(N):
        gi = g[i]
        lap = 0.0
        for d in range(6):
            q = nbf[i, d]
            if q >= 0:
                lap += g[q] - gi
        use = 0.0
        a = owner[i]
        if a >= 0:
            if cst[a] == DIV:
                dm = beta * c0
            else:
                dm = c0
            use = dm / carea[a]
        cp = Vn * gi / (Km + gi)
        take = cp if cp < use else use
        v = gi + Dg * lap + P0 * (1.0 - gi) - take
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        gn[i] = v
    for i in range(N):
        g[i] = gn[i]

    # ---- 2. aging -> shedding ----
    for cid in range(next_id):
        if calive[cid] == 0:
            continue
        if cst[cid] == SHED:
            cshed[cid] -= 1
            if cshed[cid] <= 0:
                cst[cid] = 0
            continue
        cage[cid] += dt_h
        if clin[cid] == 1:      # reverted lineage does not age out
            continue
        if cst[cid] != DIV and cage[cid] >= clife[cid]:
            cst[cid] = SHED
            cshed[cid] = shedT

    for i in range(N):
        a = owner[i]
        if a >= 0 and calive[a] == 1 and cst[a] == 0:
            owner[i] = -1
    for cid in range(next_id):
        if calive[cid] == 1 and cst[cid] == 0:
            _kill_cell(cid, calive, cst, clin, free_ids, counters)
            counters[C_SHED] += 1

    # ---- 3. claim / spreading (sequential, as in the JS stand) ----
    for i in range(N):
        if owner[i] >= 0:
            continue
        if _rnd(counters) > kClaim:
            continue
        best = -1
        ba = 1 << 30
        for d in range(6):
            q = nbf[i, d]
            if q < 0:
                continue
            a = owner[q]
            if a < 0 or cst[a] == SHED:
                continue
            if carea[a] < aMax and carea[a] < ba:
                ba = carea[a]
                best = a
        if best >= 0:
            owner[i] = best
            carea[best] += 1

    # ---- 4. shift / tension redistribution (sequential) ----
    for _t in range(N):
        i = int(_rnd(counters) * N)
        if i >= N:
            i = N - 1
        a = owner[i]
        if a < 0 or cst[a] == SHED:
            continue
        dsel = int(_rnd(counters) * 6)
        if dsel >= 6:
            dsel = 5
        j = nbf[i, dsel]
        if j < 0:
            continue
        b = owner[j]
        if b < 0 or b == a or cst[b] == SHED or carea[b] >= aMax:
            continue
        if carea[a] > carea[b]:
            keep = 0
            for d in range(6):
                q = nbf[i, d]
                if q >= 0 and owner[q] == a:
                    keep += 1
            if keep >= 1:
                owner[i] = b
                carea[a] -= 1
                carea[b] += 1

    # ---- 5. tension / neighbour census ----
    for cid in range(next_id):
        tsum[cid] = 0.0
        tcnt[cid] = 0
        cup[cid] = 0.0
        n_same[cid] = 0
        n_other[cid] = 0
        n_free_a[cid] = 0

    for i in range(N):
        a = owner[i]
        if a < 0:
            continue
        tsum[a] += carea[a]
        tcnt[a] += 1
        for d in range(6):
            q = nbf[i, d]
            if q < 0:                     # field edge is not empty space
                continue
            b = owner[q]
            if b < 0:
                tsum[a] += aMax
                tcnt[a] += 1
                n_free_a[a] += 1
            elif b != a:
                tsum[a] += carea[b]
                tcnt[a] += 1
                if cst[b] != SHED:
                    if clin[b] == 1:
                        n_same[a] += 1
                    else:
                        n_other[a] += 1
        if cst[a] == DIV:
            share = beta * c0 / carea[a]
            cp = Vn * g[i] / (Km + g[i])
            cup[a] += cp if cp < share else share

    # ---- 6. per-cell state transitions ----
    n_split = 0
    n_dying = 0
    for cid in range(next_id):
        if calive[cid] == 0 or cst[cid] == SHED:
            ctens[cid] = 1.0
            continue
        if tcnt[cid] > 0:
            tens = tsum[cid] / tcnt[cid]
        else:
            tens = 1.0
        ctens[cid] = tens

        if clin[cid] == 1:
            surrounded = (n_free_a[cid] == 0 and n_other[cid] == 0
                          and n_same[cid] > 0)
            if surrounded:
                if counters[C_REVEST] < 0:
                    counters[C_REVEST] = counters[C_STEP]
                ctol[cid] = tol_steps
                if cst[cid] != RESV and cst[cid] != DIV:
                    cst[cid] = RESV
                    cprog[cid] = 0.0
            else:
                room = (carea[cid] >= 2) or (n_free_a[cid] > 0)
                if room:
                    ctol[cid] = tol_steps
                    if cst[cid] != DIV:
                        cst[cid] = DIV
                        cprog[cid] = 0.0
                        continue      # defer substrate test to next step
                else:
                    ctol[cid] -= 1
                    if ctol[cid] <= 0:
                        dying[n_dying] = cid
                        n_dying += 1
                        counters[C_REVLOST] += 1
                        continue

        if cst[cid] == RESV:
            if tens >= Twake and carea[cid] >= 2:
                cst[cid] = DIV
                cprog[cid] = 0.0
        elif cst[cid] == DIV:
            dm = beta * c0
            ratio = cup[cid] / dm
            short = dm - cup[cid]
            if ratio < arrTh and cprog[cid] < commit:
                cst[cid] = RESV
                cprog[cid] = 0.0
            else:
                if short > 1e-12:
                    cgly[cid] -= short
                    if cgly[cid] <= 0.0:
                        cst[cid] = SHED
                        cshed[cid] = shedT
                        counters[C_STARVE] += 1
                        continue
                cprog[cid] += ratio / Tcyc
                if cprog[cid] >= 1.0 and carea[cid] >= 2:
                    split_to[cid] = _new_cell(calive, carea, cprog, cshed,
                                              split_to, free_ids, counters)
                    n_split += 1

        if cst[cid] != DIV:
            if clin[cid] == 1:
                mx = sMax0 * revStore
            else:
                mx = sMax0
            if cgly[cid] < mx:
                v = cgly[cid] + 0.02 * mx
                cgly[cid] = mx if v > mx else v

    # ---- 7. division: split the territory ----
    if n_split > 0:
        next_id = counters[C_NEXTID]
        for cid in range(next_id):
            assigned[cid] = 0
        for i in range(N):
            a = owner[i]
            if a < 0:
                continue
            kid = split_to[a]
            if kid < 0:
                continue
            if assigned[a] < (carea[a] >> 1):
                owner[i] = kid
                assigned[a] += 1
        for cid in range(next_id):
            kid = split_to[cid]
            if kid < 0:
                continue
            split_to[cid] = -1
            moved = assigned[cid]
            carea[kid] = moved
            carea[cid] -= moved
            cgly[kid] = cgly[cid] * 0.5
            cgly[cid] *= 0.5
            cage[kid] = 0.0
            cprog[kid] = 0.0
            if clin[cid] == 1:
                # parent was already reverted lineage: the new daughter
                # rolls the O2 filter independently (birth-filter mechanism)
                if _rnd(counters) < oxyRev:
                    clin[kid] = 1
                    cst[kid] = DIFF
                    clife[kid] = 1e9
                    ctol[kid] = tol_steps
                else:
                    clin[kid] = 0
                    cst[kid] = SHED
                    cshed[kid] = shedT
                    counters[C_REVOXYDEATH] += 1
                cst[cid] = DIFF
                cprog[cid] = 0.0
                cage[cid] = 0.0
            elif _rnd(counters) < pRevert:
                # fresh reversion event: this daughter also rolls the O2 filter
                counters[C_REVTOT] += 1
                if _rnd(counters) < oxyRev:
                    clin[kid] = 1
                    cst[kid] = DIFF
                    clife[kid] = 1e9
                    ctol[kid] = tol_steps
                    cgly[kid] = sMax0 * revStore
                else:
                    clin[kid] = 0
                    cst[kid] = SHED
                    cshed[kid] = shedT
                    counters[C_REVOXYDEATH] += 1
                cst[cid] = RESV
                cprog[cid] = 0.0
                cage[cid] = 0.0
            else:
                clin[kid] = 0
                if _rnd(counters) < pResv:
                    cst[kid] = RESV
                    clife[kid] = resvLifeH
                else:
                    cst[kid] = DIFF
                    clife[kid] = lifeH + (_rnd(counters) - 0.5) * 2.0 * lifeSp
                cst[cid] = RESV
                cprog[cid] = 0.0
                cage[cid] = 0.0
            counters[C_DIV] += 1

    # ---- 8. remove reverted cells that lost their niche ----
    if n_dying > 0:
        for k in range(n_dying):
            dy_flag[dying[k]] = 1
        for i in range(N):
            a = owner[i]
            if a >= 0 and dy_flag[a] == 1:
                owner[i] = -1
        for k in range(n_dying):
            cid = dying[k]
            dy_flag[cid] = 0
            if calive[cid] == 1:
                _kill_cell(cid, calive, cst, clin, free_ids, counters)


@njit(cache=True)
def _largest_clone(owner, clin, nbf, seen_n, stack_n, seen_c):
    for i in range(N):
        seen_n[i] = 0
    best = 0
    for s0 in range(N):
        a0 = owner[s0]
        if seen_n[s0] == 1 or a0 < 0 or clin[a0] != 1:
            continue
        sp = 0
        cells = 0
        stack_n[sp] = s0
        sp += 1
        seen_n[s0] = 1
        n_touched = 0
        while sp > 0:
            sp -= 1
            i = stack_n[sp]
            a = owner[i]
            if seen_c[a] == 0:
                seen_c[a] = 1
                stack_n[N - 1 - n_touched] = a   # remember to clear later
                n_touched += 1
                cells += 1
            for d in range(6):
                q = nbf[i, d]
                if q < 0 or seen_n[q] == 1:
                    continue
                b = owner[q]
                if b < 0 or clin[b] != 1:
                    continue
                seen_n[q] = 1
                stack_n[sp] = q
                sp += 1
        for k in range(n_touched):
            seen_c[stack_n[N - 1 - k]] = 0
        if cells > best:
            best = cells
    return best


class Sim:
    """Same API as epithelium_headless.Sim, so batch scripts just swap the import."""

    def __init__(self, seed, params):
        self.seed = seed
        self.p = dict(
            lifeD=6.0, sprD=3.0, pResv=1 / 11, cycH=15.0, aMax=4,
            Twake=1.08, sigma=.85, defect=600, f0=.10,
            pRevert=1 / 1000, revTolH=2.4,
            revStore=.33,   # fixed default (was .2 in the old engine)
            oxyRev=1.0,     # O2 filter: 1.0 = no filter, old behaviour
        )
        self.p.update(params)
        self._derive()
        self.reset()

    def _derive(self):
        p = self.p
        self.c0 = K['P0'] / (K['beta'] * p['sigma'])
        self.Vn = K['gamma'] * K['beta'] * self.c0
        self.Tcyc = p['cycH'] * 60 / K['dtMin']
        self.lifeH = p['lifeD'] * 24
        self.lifeSp = p['sprD'] * 24
        self.resvLifeH = p['lifeD'] * 1.3 * 24
        self.sMax0 = K['resH'] * 60 / K['dtMin'] * self.c0

    def starve_thr(self):
        return K['Km'] / (K['gamma'] - 1)

    def need_resv(self):
        return self.p['cycH'] / (self.p['lifeD'] * 24)

    def tol_steps(self):
        return max(1, int(round(self.p['revTolH'] * 60 / K['dtMin'])))

    # ---------------- state ----------------
    def reset(self):
        self.owner = np.arange(N, dtype=np.int32)
        self.cst = np.zeros(CAP, dtype=np.uint8)
        self.carea = np.zeros(CAP, dtype=np.int32)
        self.cgly = np.zeros(CAP, dtype=np.float64)
        self.cprog = np.zeros(CAP, dtype=np.float64)
        self.cage = np.zeros(CAP, dtype=np.float64)
        self.clife = np.zeros(CAP, dtype=np.float64)
        self.cshed = np.zeros(CAP, dtype=np.int32)
        self.calive = np.zeros(CAP, dtype=np.uint8)
        self.clin = np.zeros(CAP, dtype=np.uint8)
        self.ctol = np.zeros(CAP, dtype=np.int32)
        self.ctens = np.ones(CAP, dtype=np.float64)
        self.g = np.ones(N, dtype=np.float64)
        self.gn = np.zeros(N, dtype=np.float64)
        self.free_ids = np.zeros(CAP, dtype=np.int32)

        # scratch
        self.tsum = np.zeros(CAP, dtype=np.float64)
        self.tcnt = np.zeros(CAP, dtype=np.int64)
        self.cup = np.zeros(CAP, dtype=np.float64)
        self.n_same = np.zeros(CAP, dtype=np.int64)
        self.n_other = np.zeros(CAP, dtype=np.int64)
        self.n_free_a = np.zeros(CAP, dtype=np.int64)
        self.split_to = -np.ones(CAP, dtype=np.int32)
        self.assigned = np.zeros(CAP, dtype=np.int32)
        self.dying = np.zeros(CAP, dtype=np.int32)
        self.dy_flag = np.zeros(CAP, dtype=np.uint8)
        self._seen_n = np.zeros(N, dtype=np.uint8)
        self._stack_n = np.zeros(N, dtype=np.int32)
        self._seen_c = np.zeros(CAP, dtype=np.uint8)

        self.counters = np.zeros(11, dtype=np.int64)   # was 10, now 11 (+ C_REVOXYDEATH)
        self.counters[C_NEXTID] = N
        self.counters[C_REVEST] = -1
        self.counters[C_RNG] = self.seed & 0xFFFFFFFF

        # initial population, using the same LCG so seeds line up with JS
        p = self.p
        for i in range(N):
            self.calive[i] = 1
            self.carea[i] = 1
            self.cgly[i] = self.sMax0
            is_resv = _rnd(self.counters) < p['f0']
            if is_resv:
                self.cst[i] = RESV
                self.clife[i] = self.resvLifeH
            else:
                self.cst[i] = DIFF
                self.clife[i] = self.lifeH + (_rnd(self.counters) - .5) * 2 * self.lifeSp
            self.cage[i] = _rnd(self.counters) * self.clife[i]

    # ---------------- stepping ----------------
    @property
    def step(self):
        return int(self.counters[C_STEP])

    @property
    def rev_total(self):
        return int(self.counters[C_REVTOT])

    @property
    def rev_lost(self):
        return int(self.counters[C_REVLOST])

    @property
    def rev_oxy_death(self):
        return int(self.counters[C_REVOXYDEATH])

    @property
    def shed_total(self):
        return int(self.counters[C_SHED])

    @property
    def starve_total(self):
        return int(self.counters[C_STARVE])

    @property
    def div_total(self):
        return int(self.counters[C_DIV])

    def advance(self):
        p = self.p
        advance_step(
            self.owner, self.cst, self.carea, self.cgly, self.cprog,
            self.cage, self.clife, self.cshed, self.calive, self.clin,
            self.ctol, self.ctens, self.g, self.gn, NBF, self.free_ids,
            self.counters, self.tsum, self.tcnt, self.cup, self.n_same,
            self.n_other, self.n_free_a, self.split_to, self.assigned,
            self.dying, self.dy_flag,
            int(p['aMax']), float(p['Twake']), float(p['pResv']),
            float(p['pRevert']), float(p['revStore']), float(p['oxyRev']),
            self.c0, self.Vn, self.Tcyc, self.lifeH, self.lifeSp,
            self.resvLifeH, self.sMax0, self.tol_steps(),
        )
        return self.count()

    # ---------------- wound ----------------
    def inflict_wound(self, wander):
        p = self.p
        R = np.sqrt(p['defect'] * .866 / np.pi)
        if wander:
            cx = R + _rnd(self.counters) * (W - 2 * R)
            cy = (R + _rnd(self.counters) * (H - 2 * R)) * .866
        else:
            cx = (W - 1) / 2 + .25
            cy = (H - 1) / 2 * .866
        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        x = cols + (rows & 1) * .5
        y = rows * .866
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= R * R
        idxs = np.flatnonzero(mask.reshape(-1))
        owned = idxs[self.owner[idxs] >= 0]
        ids = self.owner[owned]
        self.owner[owned] = -1
        uniq, cnts = np.unique(ids, return_counts=True)
        self.carea[uniq] -= cnts
        for cid in uniq[self.carea[uniq] <= 0]:
            cid = int(cid)
            self.calive[cid] = 0
            self.cst[cid] = 0
            self.clin[cid] = 0
            self.free_ids[self.counters[C_NFREE]] = cid
            self.counters[C_NFREE] += 1
        return len(owned)

    # ---------------- readouts ----------------
    def burden(self):
        alive = self.calive.astype(bool)
        return int((alive & (self.clin == 1) & (self.cst != SHED)).sum())

    def largest_clone(self):
        return int(_largest_clone(self.owner, self.clin, NBF,
                                  self._seen_n, self._stack_n, self._seen_c))

    def count(self):
        owner = self.owner
        free = int((owner < 0).sum())
        gmin = float(self.g.min())
        ids = np.flatnonzero(self.calive.astype(bool))
        st = self.cst[ids]
        lin = self.clin[ids]
        area = self.carea[ids]
        nd = int(((lin == 0) & (st == RESV)).sum())
        nf = int(((lin == 0) & (st == DIFF)).sum())
        nv = int(((lin == 0) & (st == DIV)).sum())
        ns = int(((lin == 0) & (st != RESV) & (st != DIFF) & (st != DIV)).sum())
        rev = int(((lin == 1) & (st != RESV) & (st != SHED)).sum())
        rev_r = int(((lin == 1) & (st == RESV)).sum())
        cells = int(ids.size)
        return dict(nd=nd, nf=nf, nv=nv, ns=ns, free=free, gmin=gmin,
                    cells=cells, aMean=float(area.mean()) if cells else 1.0,
                    rev=rev, revR=rev_r, norm=nd + nf + nv + ns,
                    dens=cells / N, mob=(nv / (nd + nv)) if (nd + nv) else 0.0)


# ============================================================
# BATCH RUNNER (pilot_batch_draft_1708.py, unchanged logic —
# only the returned row dict gained revStore / oxyRev / rev_oxy_death)
# ============================================================

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
        revStore=params.get('revStore', .33),
        oxyRev=params.get('oxyRev', 1.0),
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
        rev_oxy_death=sim.rev_oxy_death,
        shed_total=sim.shed_total, div_total=sim.div_total,
    )


# ============================================================
# VALIDATION — run this first (1 seed, short horizon) before the series
# ============================================================
VALIDATE = True   # <- flip to False once the printed numbers check out

if __name__ == '__main__' and VALIDATE:
    if not HAVE_NUMBA:
        print("*** numba not installed — this will be slow. "
              "`pip install numba` first for the real thing, but the "
              "pure-Python fallback is fine for this short check. ***")

    VAL_HOURS = 2000
    VAL_SEED = 20260727   # same seed used for the JS cross-check on the father's side
    print(f"Validation run: revStore=0.33, oxyRev=0.4, seed={VAL_SEED}, "
          f"{VAL_HOURS}h, scenario='bounded', wound_start_h=1000, "
          f"wound_freq_h=24, wound_size=200")
    t0 = time.time()
    row = run_one(
        run_id=0, params=dict(lifeD=6.0, revStore=0.33, oxyRev=0.4),
        seed=VAL_SEED, target_hours=VAL_HOURS, scenario='bounded',
        wound_start_h=1000, wound_freq_h=24, wound_size=200,
    )
    dt = time.time() - t0
    print(f"done in {dt:.1f}s -> "
          f"rev_total={row['rev_total']}  rev_oxy_death={row['rev_oxy_death']}  "
          f"final_density={row['final_density']:.4f}  "
          f"final_clone_size={row['final_clone_size']}  "
          f"wounds_inflicted={row['wounds_inflicted']}  "
          f"clone_established_at_h={row['clone_established_at_h']}")
    print("\nSend these five numbers (rev_total, rev_oxy_death, "
          "final_density, final_clone_size) back for cross-check against "
          "the JS bench on the same seed before running the full series.")
    raise SystemExit


# ============================================================
# FULL SERIES — only runs once VALIDATE = False above
#
# This series is the revStore=0.33 BASELINE with the O2 filter switched off
# (oxyRev=1.0), across all three scenarios: 'none', 'bounded', 'chronic'.
# Purpose: chapters 1-2 currently compare their revStore=0.33 results
# against an OLDER run that used full (untruncated) glycogen — this series
# replaces that comparison point with the correct, controlled one (same
# revStore as everywhere else in chapters 1-2, oxygen filter off).
# ============================================================
if __name__ == '__main__':
    # 100 seeds per (revStore, scenario) combination, grid = [0.20, 0.25, 0.30]
    # x scenarios = ['bounded', 'chronic'] -> 600 runs total. 'none' is
    # deliberately NOT included here: the revStore=0.33 baseline already
    # showed 100% establishment at 'none' with the filter off, and going
    # to a lower glycogen without any wound would only make that more true
    # — not an interesting question. This series is specifically about
    # whether a lower glycogen floor changes anything once there's a wound.
    TARGET_HOURS = 20000
    STOP_THRESHOLD = 500
    OXY_REV = 1.0          # filter off — same no-filter condition as the 0.33 baseline
    REV_STORE_GRID = [0.20, 0.25, 0.30]
    SCENARIOS = ['bounded', 'chronic']
    N_SEEDS = 100

    # Wound protocol — identical to every other series in this file.
    WOUND_START_H = 1000
    WOUND_SIZE = 200
    WOUND_FREQ_CHOICES = [12, 24, 48]
    FREQ_DRAW_SEED = 12345   # fixed, so the freq draw is reproducible

    # NEW file — this one varies revStore (not oxyRev), unlike every other
    # grid series in this project so far.
    OUT_CSV = 'runs_series_glycogen_grid_wound_2026-09.csv'

    SESSION_MINUTES = 30

    if not HAVE_NUMBA:
        print("*** WARNING: numba is NOT installed — the engine will run in "
              "pure-Python fallback mode, which is far SLOWER than the old "
              "NumPy version, let alone the compiled one. Run "
              "`!pip install numba` first. ***")

    N_WORKERS = os.cpu_count() or 1
    print(f"Using {N_WORKERS} worker processes (os.cpu_count()).")

    # ---- what's already done? (resume) ----
    # Key is (revStore, scenario, seed) — revStore now varies within this
    # series, so it has to be part of the resume key.
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline='') as fh:
            for r in csv.DictReader(fh):
                done.add((round(float(r['revStore']), 6), r['scenario'], int(r['seed'])))
        print(f"found {len(done)} completed runs in {OUT_CSV} — skipping those")

    # run_id derived from (revStore index, scenario index, seed) so it stays
    # stable if a later session runs only a subset of the grid. The real
    # key for joining in pandas is (revStore, scenario, seed), not run_id.
    jobs = []
    freq_rng = random.Random(FREQ_DRAW_SEED)
    for rs_idx, rev_store in enumerate(REV_STORE_GRID):
        for scen_idx, scenario in enumerate(SCENARIOS):
            for seed in range(N_SEEDS):
                key = (round(rev_store, 6), scenario, seed)
                wound_freq_h = freq_rng.choice(WOUND_FREQ_CHOICES)
                if key in done:
                    continue
                run_id = rs_idx * 10000000 + scen_idx * 1000000 + seed
                jobs.append(dict(
                    run_id=run_id,
                    params=dict(lifeD=6.0, revStore=rev_store, oxyRev=OXY_REV),
                    seed=seed,
                    target_hours=TARGET_HOURS, scenario=scenario,
                    wound_start_h=WOUND_START_H,
                    wound_freq_h=wound_freq_h,
                    wound_size=WOUND_SIZE,
                    stop_threshold=STOP_THRESHOLD,
                ))

    total = len(REV_STORE_GRID) * len(SCENARIOS) * N_SEEDS
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
