"""
run_batch_missing.py — the runs still missing for the article
on the glycogen x oxygen axis (Part I, Chapters 1-2; Appendix A.2-A.3).  Prepared 2026-09-28.

ENGINE: byte-for-byte the same engine as run_batch_oxygrid.py and
run_batch_oxygrid_wound.py (epithelium__numba.py + oxyRev filter), same LCG,
same RNG call order.  Nothing in the engine was changed.

WHAT IS NEW IN THE RUNNER (not in the engine):
  1. Trajectory logging.  The largest clone and the reverted burden are
     recorded every 24 h for the whole run (not only until the threshold),
     into a second CSV.  This lets us change the establishment threshold
     (250 / 500 / 1000) AFTER the fact, without re-running anything.
     Caveat: in 'bounded' the threshold (500) also switches wounding off,
     so for bounded only the 500 threshold is protocol-exact.
  2. Paired wound interval.  wound_freq_h is now a deterministic function
     of the seed, so bounded and chronic runs of the SAME seed get the SAME
     interval.  (In the earlier wound series they were drawn independently.)
  3. One script for every series: pick the series with SERIES below or
     with the command line:  python run_batch_missing.py S1

OUTPUT: one pair of files per series, e.g.
    missing_S1_runs.csv   — one row per run (same columns as before + a few)
    missing_S1_traj.csv   — one row per run: key columns + traj_clone,
                            traj_burden (semicolon-separated, every 24 h)
Resume works exactly as before: finished runs are skipped on relaunch.

    pip install numba      # in DataLab, if not already installed
"""
import os
import sys
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
# RUNNER — same logic as before + trajectory logging, paired wound
# interval, optional random wound position.
# ============================================================

WOUND_START_H = 1000
WOUND_SIZE = 200
WOUND_FREQ_CHOICES = [12, 24, 48]
FREQ_DRAW_SEED = 12345


def paired_wound_freq(seed):
    """Same interval for bounded and chronic runs of the same seed."""
    return random.Random(FREQ_DRAW_SEED * 1000003 + seed).choice(WOUND_FREQ_CHOICES)


def run_one(series, params, seed, target_hours, scenario, wander=False,
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
    check_every_steps = max(1, int(round(check_every_h * 60 / K['dtMin'])))
    traj_clone, traj_burden = [], []

    while sim.step < steps_target:
        due = (wounding_active and wound_start_step is not None
               and sim.step >= wound_start_step
               and (last_wound_step is None
                    or sim.step - last_wound_step >= wound_freq_steps))
        if due:
            sim.inflict_wound(wander)
            wounds_inflicted += 1
            last_wound_step = sim.step
        sim.advance()

        if sim.step % check_every_steps == 0:
            lc = sim.largest_clone()
            traj_clone.append(lc)
            traj_burden.append(sim.burden())
            if clone_established_at_h is None and lc > stop_threshold:
                clone_established_at_h = sim.step * K['dtMin'] / 60
                if scenario == 'bounded':
                    wounding_active = False   # permanent stop for this run

    final_clone = sim.largest_clone()
    final_count = sim.count()
    row = dict(
        series=series, scenario=scenario, seed=seed,
        lattice=f'{W}x{H}', wander=int(wander),
        tau_d=p.get('lifeD'), cyc_h=p.get('cycH', 15.0),
        Twake=p.get('Twake', 1.08), sigma=p.get('sigma', .85),
        pRevert=p.get('pRevert', 1 / 1000),
        revStore=p.get('revStore', .33), oxyRev=p.get('oxyRev', 1.0),
        wound_start_h=wound_start_h, wound_freq_h=wound_freq_h,
        wound_size=wound_size, wounds_inflicted=wounds_inflicted,
        stop_threshold=stop_threshold, check_every_h=check_every_h,
        wounding_stops_on_threshold=(scenario == 'bounded'),
        clone_established_at_h=clone_established_at_h,
        target_hours=target_hours, elapsed_h=sim.step * K['dtMin'] / 60,
        final_clone_size=final_clone,
        final_cells_total=final_count['cells'],
        final_density=final_count['dens'],
        final_clone_frac=(final_clone / final_count['cells']
                          if final_count['cells'] else 0.0),
        final_burden=sim.burden(),
        rev_total=sim.rev_total, rev_lost=sim.rev_lost,
        rev_oxy_death=sim.rev_oxy_death, starve_total=sim.starve_total,
        shed_total=sim.shed_total, div_total=sim.div_total,
    )
    traj = {k: row[k] for k in KEY_COLS}
    traj['traj_clone'] = ';'.join(map(str, traj_clone))
    traj['traj_burden'] = ';'.join(map(str, traj_burden))
    return row, traj


KEY_COLS = ['series', 'scenario', 'lattice', 'wander', 'pRevert',
            'revStore', 'oxyRev', 'seed']


def key_of(d):
    return (d['series'], d['scenario'], str(d['lattice']), int(d['wander']),
            round(float(d['pRevert']), 8), round(float(d['revStore']), 6),
            round(float(d['oxyRev']), 6), int(d['seed']))


# ============================================================
# SERIES DEFINITIONS
# Each entry expands into a list of jobs.  Seeds are always 0..n_seeds-1
# (paired design across every grid, as in all earlier series).
# ============================================================
OXY_FULL = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]


def grid(series, revs, oxys, scenarios, n_seeds=100, pRevert=1 / 1000,
         wander=False, target_hours=20000):
    jobs = []
    for sc in scenarios:
        for r in revs:
            for o in oxys:
                for s in range(n_seeds):
                    wounded = sc in ('bounded', 'chronic')
                    jobs.append(dict(
                        series=series, seed=s, scenario=sc, wander=wander,
                        target_hours=target_hours,
                        params=dict(lifeD=6.0, revStore=r, oxyRev=o,
                                    pRevert=pRevert),
                        wound_start_h=WOUND_START_H if wounded else None,
                        wound_freq_h=paired_wound_freq(s) if wounded else None,
                        wound_size=WOUND_SIZE if wounded else None,
                    ))
    return jobs


SERIES_DEF = {
    # S0 — reproducibility check against the OLD csv files (run first!)
    #      5 seeds each; must match old rows exactly (see task sheet).
    'S0': lambda: (grid('S0', [0.33], [0.40], ['none'], n_seeds=5)
                   + grid('S0', [0.33], [1.0], ['none'], n_seeds=5)),
    # S1 — the missing corner: full glycogen with the O2 filter,
    #      + the upper end of the O2 grid at revStore 0.33.   (1000 runs)
    'S1': lambda: (grid('S1', [1.0], OXY_FULL, ['none'])
                   + grid('S1', [0.33], [0.55, 0.60], ['none'])),
    # S2 — rest of the revStore x oxyRev phase diagram, no wound (1600 runs)
    'S2': lambda: grid('S2', [0.50, 0.75], OXY_FULL, ['none']),
    # S3 — glycogen under wounding at oxyRev 0.5, paired bounded/chronic
    #      (1000 runs)
    'S3': lambda: grid('S3', [0.20, 0.25, 0.33, 0.50, 1.0], [0.50],
                       ['bounded', 'chronic']),
}

SESSION_MINUTES = 30      # stop launching new runs after this; None = no limit


def main(series):
    if not HAVE_NUMBA:
        print("*** WARNING: numba is NOT installed — pure-Python fallback "
              "is far too slow for the real series. `pip install numba`. ***")

    runs_csv, traj_csv = f'missing_{series}_runs.csv', f'missing_{series}_traj.csv'
    jobs = SERIES_DEF[series]()
    for j in jobs:
        j['_key'] = key_of(dict(series=j['series'], scenario=j['scenario'],
                                lattice=f'{W}x{H}', wander=j['wander'],
                                pRevert=j['params']['pRevert'],
                                revStore=j['params']['revStore'],
                                oxyRev=j['params']['oxyRev'], seed=j['seed']))
    done = set()
    if os.path.exists(runs_csv):
        with open(runs_csv, newline='') as fh:
            done = {key_of(r) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if j['_key'] not in done]
    print(f"series {series}: {len(jobs)} runs total, {len(done)} done, "
          f"{len(todo)} to go; lattice {W}x{H}; "
          f"{os.cpu_count()} workers")
    if not todo:
        print("nothing to do — the series is complete.")
        return

    t0 = time.time()
    n_workers = os.cpu_count() or 1
    new_runs = not os.path.exists(runs_csv) or os.path.getsize(runs_csv) == 0
    new_traj = not os.path.exists(traj_csv) or os.path.getsize(traj_csv) == 0
    fr = open(runs_csv, 'a', newline='')
    ft = open(traj_csv, 'a', newline='')
    wr = wt = None
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for start in range(0, len(todo), n_workers):
                if (SESSION_MINUTES is not None
                        and (time.time() - t0) / 60 >= SESSION_MINUTES):
                    break
                wave = todo[start:start + n_workers]
                futs = [ex.submit(run_one, **{k: v for k, v in j.items()
                                              if k != '_key'}) for j in wave]
                for fut in as_completed(futs):
                    row, traj = fut.result()
                    if wr is None:
                        wr = csv.DictWriter(fr, fieldnames=list(row))
                        if new_runs:
                            wr.writeheader()
                        wt = csv.DictWriter(ft, fieldnames=list(traj))
                        if new_traj:
                            wt.writeheader()
                    wr.writerow(row); fr.flush()
                    wt.writerow(traj); ft.flush()
                    completed += 1
                    print(f"[{completed}/{len(todo)}] {row['scenario']} "
                          f"rev={row['revStore']} oxy={row['oxyRev']} "
                          f"seed={row['seed']} -> clone={row['final_clone_size']} "
                          f"est_at={row['clone_established_at_h']} "
                          f"({(time.time() - t0) / 60:.1f} min)")
    finally:
        fr.close(); ft.close()
    left = len(todo) - completed
    print(f"\n{completed} runs this session in {(time.time() - t0) / 60:.1f} min"
          f" -> {runs_csv}, {traj_csv}")
    print(f"{left} still to go — run the same command again to continue."
          if left else "series complete.")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'S0')
