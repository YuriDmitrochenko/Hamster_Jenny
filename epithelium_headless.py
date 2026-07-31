"""
Headless (no-graphics) port of the epithelium territory-model bench.

Faithful to the JS stand for:
  - cell territories (1..aMax nodes), tension = mean area over self+neighbours
  - RESV / DIFF / DIV / SHED state machine, division splits territory in half
  - background stochastic reversion to an undifferentiated lineage,
    niche-dependent survival (surrounded -> reserve, room -> divide, else dies)
  - reserve-sufficiency feasibility logic
  - wound infliction

Deliberate approximation (documented, not hidden):
  - "spreading" (claim) and "shift" (tension redistribution) are applied
    SYNCHRONOUSLY (all trials computed from the same pre-step snapshot,
    then applied together) instead of the JS's one-node-at-a-time random
    sequential update. This is necessary to get acceptable performance in
    pure Python/NumPy (no JIT available in this environment). It preserves
    the direction and rough magnitude of both processes (free nodes still
    go to the least-stretched neighbour; area still flows from more- to
    less-stretched cells) but is not bit-identical to the visual stand.
    -> Recommended: run a short (e.g. 500 h) side-by-side comparison
       against the JS stand before trusting large batches on this port.
"""
import numpy as np
import pandas as pd
import csv
import time
import argparse
import sys

W, H = 76, 79
N = W * H
CAP = N + 8

RESV, DIFF, DIV, SHED = 1, 2, 3, 4

K = dict(P0=.20, Dg=.15, Km=.40, gamma=1.6, dtMin=3, commit=.40, arrTh=.75,
         shedT=8, kClaim=.5, resH=25, beta=12)


def build_neighbors():
    """NBF[i, d] = node index of neighbour d of node i, or -1 at the edge."""
    NBF = -np.ones((N, 6), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            i = r * W + c
            odd = r & 1
            if odd:
                T = [(1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)]
            else:
                T = [(1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
            for d, (dx, dy) in enumerate(T):
                nc, nr = c + dx, r + dy
                if 0 <= nc < W and 0 <= nr < H:
                    NBF[i, d] = nr * W + nc
    return NBF


NBF = build_neighbors()
NBF_FLAT = NBF.reshape(-1)          # (N*6,) with -1 for missing
VALID_FLAT = NBF_FLAT >= 0
SRC_FLAT = np.repeat(np.arange(N), 6)  # source node for each of the N*6 slots


class Sim:
    def __init__(self, seed, params):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.p = dict(
            lifeD=6.0, sprD=3.0, pResv=1 / 11, cycH=15.0, aMax=4,
            Twake=1.08, sigma=.85, defect=600, f0=.10,
            pRevert=1 / 1000, revTolH=2.4, revStore=.2, woundEvery=0,
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
        return max(1, round(self.p['revTolH'] * 60 / K['dtMin']))

    # ---------------- state -----------------
    def reset(self):
        rng = self.rng
        self.owner = np.arange(N, dtype=np.int32)   # node i -> cell id i initially
        cap = CAP
        self.cst = np.zeros(cap, dtype=np.uint8)
        self.carea = np.zeros(cap, dtype=np.int32)
        self.cgly = np.zeros(cap, dtype=np.float64)
        self.cprog = np.zeros(cap, dtype=np.float64)
        self.cage = np.zeros(cap, dtype=np.float64)
        self.clife = np.zeros(cap, dtype=np.float64)
        self.cshed = np.zeros(cap, dtype=np.int16)
        self.calive = np.zeros(cap, dtype=np.uint8)
        self.clin = np.zeros(cap, dtype=np.uint8)     # 0 normal, 1 reverted
        self.ctol = np.zeros(cap, dtype=np.int32)
        self.free_ids = []
        self.next_id = N

        f0 = self.p['f0']
        is_resv = rng.random(N) < f0
        self.cst[:N] = np.where(is_resv, RESV, DIFF)
        self.calive[:N] = 1
        self.carea[:N] = 1
        self.cgly[:N] = self.sMax0
        spread = (rng.random(N) - .5) * 2 * self.lifeSp
        self.clife[:N] = np.where(is_resv, self.resvLifeH, self.lifeH + spread)
        self.cage[:N] = rng.random(N) * self.clife[:N]

        self.step = 0
        self.shed_total = 0
        self.starve_total = 0
        self.div_total = 0
        self.rev_total = 0
        self.rev_lost = 0
        self.rev_established = False
        self.rev_est_step = -1
        self.wound_at = -1
        self.wound_n = 0
        self.wound_base = 0
        self.last_wound = -10 ** 9
        self.wound_count = 0

        self.g = np.ones(N, dtype=np.float64)

        self.clone_init = None
        self.clone_max = 0
        self.clone_final = None

    def new_cell(self):
        if self.free_ids:
            cid = self.free_ids.pop()
        else:
            cid = self.next_id
            self.next_id += 1
        self.calive[cid] = 1
        self.carea[cid] = 0
        self.cprog[cid] = 0
        self.cshed[cid] = 0
        return cid

    def kill_cell(self, cid):
        self.calive[cid] = 0
        self.cst[cid] = 0
        self.clin[cid] = 0
        self.free_ids.append(cid)

    # ---------------- wound -----------------
    def inflict_wound(self, wander):
        rng = self.rng
        c = self.count()
        self.wound_base = c['free']
        p = self.p
        R = np.sqrt(p['defect'] * .866 / np.pi)
        if wander:
            cx = R + rng.random() * (W - 2 * R)
            cy = (R + rng.random() * (H - 2 * R)) * .866
        else:
            cx = (W - 1) / 2 + .25
            cy = (H - 1) / 2 * .866

        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        odd = rows & 1
        x = cols + odd * .5
        y = rows * .866
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= R * R
        idxs = np.flatnonzero(mask.reshape(-1))
        owned = idxs[self.owner[idxs] >= 0]
        ids = self.owner[owned]
        self.owner[owned] = -1
        # decrement area, kill cells that hit zero
        uniq, counts = np.unique(ids, return_counts=True)
        self.carea[uniq] -= counts
        dead = uniq[self.carea[uniq] <= 0]
        for cid in dead:
            self.kill_cell(int(cid))
        self.wound_n = len(owned)
        self.wound_at = self.step
        self.last_wound = self.step
        self.wound_count += 1

    # ---------------- one step -----------------
    def advance(self):
        p = self.p
        self.step += 1
        if p['woundEvery'] > 0 and self.step - self.last_wound >= p['woundEvery'] * 60 / K['dtMin']:
            self.inflict_wound(True)

        beta, Km, Dg, P0, aMax = K['beta'], K['Km'], K['Dg'], K['P0'], p['aMax']
        owner = self.owner

        # ---- 1. glucose diffusion + consumption ----
        g = self.g
        nb_g = g[NBF_FLAT]
        nb_g = np.where(VALID_FLAT, nb_g, g[SRC_FLAT])  # missing neighbour contributes 0 to laplacian
        lap = np.zeros(N)
        np.add.at(lap, SRC_FLAT, np.where(VALID_FLAT, nb_g - g[SRC_FLAT], 0.0))
        owned_mask = owner >= 0
        use = np.zeros(N)
        oid = owner[owned_mask]
        dm = np.where(self.cst[oid] == DIV, beta * self.c0, self.c0)
        use[owned_mask] = dm / self.carea[oid]
        cp = self.Vn * g / (Km + g)
        v = g + Dg * lap + P0 * (1 - g) - np.minimum(cp, use)
        self.g = np.clip(v, 0, 1)

        # ---- 2. aging -> shed ----
        alive = self.calive.astype(bool)
        shedding = alive & (self.cst == SHED)
        self.cshed[shedding] -= 1
        done = shedding & (self.cshed <= 0)
        self.cst[done] = 0
        aging = alive & (~shedding) & (self.clin == 0)
        self.cage[aging] += K['dtMin'] / 60
        to_shed = aging & (self.cst != DIV) & (self.cage >= self.clife)
        self.cst[to_shed] = SHED
        self.cshed[to_shed] = K['shedT']

        owned_mask = owner >= 0
        zero_state = owned_mask & (self.cst[np.where(owned_mask, owner, 0)] == 0)
        owner[zero_state] = -1
        dead_now = (self.calive == 1) & (self.cst == 0)
        shed_count = int(dead_now.sum())
        self.shed_total += shed_count
        for cid in np.flatnonzero(dead_now):
            self.kill_cell(int(cid))

        # ---- 3. claim (spreading) — synchronous approximation ----
        free_mask = owner < 0
        free_idx = np.flatnonzero(free_mask)
        if free_idx.size:
            trial = self.rng.random(free_idx.size) <= K['kClaim']
            cand_idx = free_idx[trial]
            if cand_idx.size:
                nbrs = NBF[cand_idx]                       # (n,6)
                valid_n = nbrs >= 0
                nb_owner = np.where(valid_n, owner[np.clip(nbrs, 0, None)], -1)
                nb_area = np.where(valid_n & (nb_owner >= 0), self.carea[np.clip(nb_owner, 0, None)], 10 ** 9)
                nb_shed = np.where(valid_n & (nb_owner >= 0), self.cst[np.clip(nb_owner, 0, None)] == SHED, True)
                nb_full = np.where(valid_n & (nb_owner >= 0), self.carea[np.clip(nb_owner, 0, None)] >= aMax, True)
                bad = (~valid_n) | (nb_owner < 0) | nb_shed | nb_full
                nb_area = np.where(bad, 10 ** 9, nb_area)
                best_d = np.argmin(nb_area, axis=1)
                best_area = nb_area[np.arange(nb_area.shape[0]), best_d]
                best_owner = nb_owner[np.arange(nb_owner.shape[0]), best_d]
                ok = best_area < 10 ** 9
                claim_node = cand_idx[ok]
                claim_owner = best_owner[ok]
                if claim_node.size:
                    # cap simultaneous claims per owner at its remaining capacity
                    order = self.rng.permutation(claim_node.size)
                    claim_node, claim_owner = claim_node[order], claim_owner[order]
                    room = np.maximum(0, aMax - self.carea[claim_owner])
                    # rank of each claim among claims to the same owner
                    srt = np.argsort(claim_owner, kind='stable')
                    co_sorted = claim_owner[srt]
                    rank = np.zeros_like(co_sorted)
                    if co_sorted.size:
                        change = np.empty(co_sorted.size, dtype=bool)
                        change[0] = True
                        change[1:] = co_sorted[1:] != co_sorted[:-1]
                        grp_start = np.where(change)[0]
                        rank_sorted = np.arange(co_sorted.size) - np.repeat(grp_start, np.diff(np.append(grp_start, co_sorted.size)))
                        rank[srt] = rank_sorted
                    accept = rank < room
                    acc_node = claim_node[accept]
                    acc_owner = claim_owner[accept]
                    owner[acc_node] = acc_owner
                    uniq_o, cnt_o = np.unique(acc_owner, return_counts=True)
                    self.carea[uniq_o] += cnt_o

        # ---- 4. shift (tension redistribution) — synchronous approximation ----
        alive_owned = owner >= 0
        trials = N
        i_arr = self.rng.integers(0, N, size=trials)
        d_arr = self.rng.integers(0, 6, size=trials)
        j_arr = NBF[i_arr, d_arr]
        valid = j_arr >= 0
        a_arr = np.where(valid, owner[i_arr], -1)
        bb_arr = np.where(valid, owner[np.clip(j_arr, 0, None)], -1)
        ok = valid & (a_arr >= 0) & (bb_arr >= 0) & (a_arr != bb_arr)
        ok &= np.where(ok, self.cst[np.clip(a_arr, 0, None)] != SHED, False)
        ok &= np.where(ok, self.cst[np.clip(bb_arr, 0, None)] != SHED, False)
        ok &= np.where(ok, self.carea[np.clip(bb_arr, 0, None)] < aMax, False)
        ok &= np.where(ok, self.carea[np.clip(a_arr, 0, None)] > self.carea[np.clip(bb_arr, 0, None)], False)
        # "keep>=1": donor a must retain >=1 other node among i's neighbours owned by a
        nbrs_i = NBF[i_arr]
        keep_cnt = np.zeros(trials, dtype=np.int32)
        for d in range(6):
            nb = nbrs_i[:, d]
            v2 = nb >= 0
            match = v2 & (np.where(v2, owner[np.clip(nb, 0, None)], -2) == a_arr)
            keep_cnt += match.astype(np.int32)
        ok &= (keep_cnt >= 1)
        sel_i = i_arr[ok]
        sel_a = a_arr[ok]
        sel_bb = bb_arr[ok]
        if sel_i.size:
            # Enforce at most one transfer per target node, per donor, and per
            # acceptor within this synchronous batch. Each condition (a>bb,
            # bb<aMax) was checked against the pre-step snapshot, so capping
            # every id to a single use per step keeps 0<=carea<=aMax exactly
            # (this is the fix for the donor-emptied / acceptor-overfull bug
            # that a naive synchronous batch would otherwise introduce).
            idx = np.arange(sel_i.size)
            keep = pd.Series(idx).groupby(sel_i, sort=False).head(1).index.values
            idx = idx[keep]
            keep = pd.Series(np.arange(idx.size)).groupby(sel_a[idx], sort=False).head(1).index.values
            idx = idx[keep]
            keep = pd.Series(np.arange(idx.size)).groupby(sel_bb[idx], sort=False).head(1).index.values
            idx = idx[keep]
            fi, fa, fbb = sel_i[idx], sel_a[idx], sel_bb[idx]
            owner[fi] = fbb
            self.carea[fa] -= 1
            self.carea[fbb] += 1

        # ---- 5. tension / neighbour census per cell ----
        cap = CAP
        tsum = np.zeros(cap)
        tcnt = np.zeros(cap, dtype=np.int64)
        n_same = np.zeros(cap, dtype=np.int64)
        n_other = np.zeros(cap, dtype=np.int64)
        n_free = np.zeros(cap, dtype=np.int64)
        cup = np.zeros(cap)

        owned_mask = owner >= 0
        own_idx = np.flatnonzero(owned_mask)
        oid = owner[own_idx]
        np.add.at(tsum, oid, self.carea[oid])
        np.add.at(tcnt, oid, 1)

        nbrs = NBF[own_idx]           # (n,6)
        valid_n = nbrs >= 0
        nb_owner = np.where(valid_n, owner[np.clip(nbrs, 0, None)], -2)
        is_free = valid_n & (nb_owner < 0)
        is_other_cell = valid_n & (nb_owner >= 0) & (nb_owner != oid[:, None])
        for d in range(6):
            fmask = is_free[:, d]
            if fmask.any():
                np.add.at(tsum, oid[fmask], aMax)
                np.add.at(tcnt, oid[fmask], 1)
                np.add.at(n_free, oid[fmask], 1)
            omask = is_other_cell[:, d]
            if omask.any():
                nb_o = nb_owner[omask, d]
                not_shed = self.cst[nb_o] != SHED
                np.add.at(tsum, oid[omask], self.carea[nb_o])
                np.add.at(tcnt, oid[omask], 1)
                lin_same = self.clin[nb_o] == 1
                sel_same = omask.copy()
                sel_same[omask] = not_shed & lin_same
                sel_other = omask.copy()
                sel_other[omask] = not_shed & (~lin_same)
                if sel_same.any():
                    np.add.at(n_same, oid[sel_same], 1)
                if sel_other.any():
                    np.add.at(n_other, oid[sel_other], 1)

        div_mask_nodes = owned_mask & (self.cst[np.where(owned_mask, owner, 0)] == DIV)
        if div_mask_nodes.any():
            oid_div = owner[div_mask_nodes]
            share = beta * self.c0 / self.carea[oid_div]
            cp_n = self.Vn * self.g[div_mask_nodes] / (Km + self.g[div_mask_nodes])
            np.add.at(cup, oid_div, np.minimum(cp_n, share))

        tens = np.ones(cap)
        has_cnt = tcnt > 0
        tens[has_cnt] = tsum[has_cnt] / tcnt[has_cnt]

        # ---- 6. per-cell state transitions ----
        alive_mask = self.calive.astype(bool)
        not_shed_mask = alive_mask & (self.cst != SHED)
        ids_active = np.flatnonzero(not_shed_mask)
        cst_pre = self.cst.copy()   # snapshot: each id is processed against ITS state
                                     # at the START of this step's transitions, exactly
                                     # once — mirrors the JS if/else-if per id per step
                                     # (a cell that wakes RESV->DIV this step must NOT
                                     # also be run through the DIV branch this same step)

        dying = []
        split_to = {}

        # reverted lineage niche rule
        rev_ids = ids_active[self.clin[ids_active] == 1]
        for cid in rev_ids:
            surrounded = (n_free[cid] == 0) and (n_other[cid] == 0) and (n_same[cid] > 0)
            if surrounded:
                if not self.rev_established:
                    self.rev_established = True
                    self.rev_est_step = self.step
                self.ctol[cid] = self.tol_steps()
                if self.cst[cid] not in (RESV, DIV):
                    self.cst[cid] = RESV
                    self.cprog[cid] = 0
            else:
                room = (self.carea[cid] >= 2) or (n_free[cid] > 0)
                if room:
                    self.ctol[cid] = self.tol_steps()
                    if self.cst[cid] != DIV:
                        self.cst[cid] = DIV
                        self.cprog[cid] = 0
                        continue
                else:
                    self.ctol[cid] -= 1
                    if self.ctol[cid] <= 0:
                        dying.append(cid)
                        self.rev_lost += 1
                        continue

        active2 = np.array([c for c in ids_active if c not in dying], dtype=np.int64)
        resv_ids = active2[cst_pre[active2] == RESV]
        wake = resv_ids[(tens[resv_ids] >= p['Twake']) & (self.carea[resv_ids] >= 2)]
        self.cst[wake] = DIV
        self.cprog[wake] = 0

        div_ids = active2[cst_pre[active2] == DIV]
        for cid in div_ids:
            dm = beta * self.c0
            ratio = cup[cid] / dm
            short = dm - cup[cid]
            if ratio < K['arrTh'] and self.cprog[cid] < K['commit']:
                self.cst[cid] = RESV
                self.cprog[cid] = 0
                continue
            if short > 1e-12:
                self.cgly[cid] -= short
                if self.cgly[cid] <= 0:
                    self.cst[cid] = SHED
                    self.cshed[cid] = K['shedT']
                    self.starve_total += 1
                    continue
            self.cprog[cid] += ratio / self.Tcyc
            if self.cprog[cid] >= 1 and self.carea[cid] >= 2:
                split_to[cid] = self.new_cell()

        # glycogen replenishment for non-dividing active cells
        non_div = active2[self.cst[active2] != DIV]
        mx = np.where(self.clin[non_div] == 1, self.sMax0 * p['revStore'], self.sMax0)
        under = self.cgly[non_div] < mx
        idxu = non_div[under]
        self.cgly[idxu] = np.minimum(mx[under], self.cgly[idxu] + .02 * mx[under])

        # ---- 7. division: split territory ----
        if split_to:
            assigned = {cid: 0 for cid in split_to}
            parent_of_node = owner.copy()
            for i in range(N):
                a = parent_of_node[i]
                if a in split_to:
                    half = self.carea[a] >> 1
                    if assigned[a] < half:
                        owner[i] = split_to[a]
                        assigned[a] += 1
            for cid, kid in split_to.items():
                moved = assigned[cid]
                self.carea[kid] = moved
                self.carea[cid] -= moved
                self.cgly[kid] = self.cgly[cid] * .5
                self.cgly[cid] *= .5
                self.cage[kid] = 0
                self.cprog[kid] = 0
                if self.clin[cid] == 1:
                    self.clin[kid] = 1
                    self.cst[kid] = DIFF
                    self.clife[kid] = 1e9
                    self.ctol[kid] = self.tol_steps()
                    self.cst[cid] = DIFF
                    self.cprog[cid] = 0
                    self.cage[cid] = 0
                elif self.rng.random() < p['pRevert']:
                    self.clin[kid] = 1
                    self.cst[kid] = DIFF
                    self.clife[kid] = 1e9
                    self.ctol[kid] = self.tol_steps()
                    self.cgly[kid] = self.sMax0 * p['revStore']
                    self.rev_total += 1
                    self.cst[cid] = RESV
                    self.cprog[cid] = 0
                    self.cage[cid] = 0
                else:
                    self.clin[kid] = 0
                    is_r = self.rng.random() < p['pResv']
                    self.cst[kid] = RESV if is_r else DIFF
                    spr = (self.rng.random() - .5) * 2 * self.lifeSp
                    self.clife[kid] = self.resvLifeH if is_r else self.lifeH + spr
                    self.cst[cid] = RESV
                    self.cprog[cid] = 0
                    self.cage[cid] = 0
                self.div_total += 1

        # ---- 8. remove dying reverted cells ----
        if dying:
            dying_arr = np.array(dying)
            kill_mask = np.isin(owner, dying_arr) & (owner >= 0)
            owner[kill_mask] = -1
            for cid in dying:
                if self.calive[cid]:
                    self.kill_cell(int(cid))

        c = self.count()
        if self.clone_init is None:
            self.clone_init = c['clone']
        if c['clone'] > self.clone_max:
            self.clone_max = c['clone']
        self.clone_final = c['clone']
        return c

    # ---------------- readouts -----------------
    def count(self):
        owner = self.owner
        free = int((owner < 0).sum())
        gmin = float(self.g.min()) if N else 1.0
        alive_mask = self.calive.astype(bool)
        ids = np.flatnonzero(alive_mask)
        st = self.cst[ids]
        lin = self.clin[ids]
        area = self.carea[ids]
        nd = int(((lin == 0) & (st == RESV)).sum())
        nf = int(((lin == 0) & (st == DIFF)).sum())
        nv = int(((lin == 0) & (st == DIV)).sum())
        ns = int(((lin == 0) & (st != RESV) & (st != DIFF) & (st != DIV)).sum())
        rev = int(((lin == 1) & (st != RESV) & (st != SHED)).sum())
        rev_r = int(((lin == 1) & (st == RESV)).sum())
        norm = nd + nf + nv + ns
        cells = int(ids.size)
        a_mean = float(area.mean()) if cells else 1.0
        dens = cells / N
        mob = nv / (nd + nv) if (nd + nv) else 0.0
        clone = self.largest_clone()
        return dict(nd=nd, nf=nf, nv=nv, ns=ns, free=free, gmin=gmin,
                    cells=cells, aMean=a_mean, rev=rev, revR=rev_r, norm=norm,
                    dens=dens, mob=mob, clone=clone)

    def burden(self):
        """Total number of live reverted-lineage cells anywhere in the tissue
        (not just the largest connected patch). Cheap: no BFS, just a mask sum.
        This is the metric the paper's clinical framing cares about ('area of
        involvement' / mosaic dysplasia burden), not the largest single clone."""
        alive = self.calive.astype(bool)
        rev_lineage = self.clin == 1
        not_shed = self.cst != SHED
        return int((alive & rev_lineage & not_shed).sum())

    def largest_clone(self):
        rev_mask_by_id = self.clin == 1
        owner = self.owner
        owned = owner >= 0
        node_is_rev = np.zeros(N, dtype=bool)
        node_is_rev[owned] = rev_mask_by_id[owner[owned]]
        if not node_is_rev.any():
            return 0
        seen = np.zeros(N, dtype=bool)
        best = 0
        idxs = np.flatnonzero(node_is_rev)
        for s0 in idxs:
            if seen[s0]:
                continue
            stack = [s0]
            seen[s0] = True
            cell_ids = set()
            while stack:
                i = stack.pop()
                cell_ids.add(int(owner[i]))
                for nb in NBF[i]:
                    if nb < 0 or seen[nb] or not node_is_rev[nb]:
                        continue
                    seen[nb] = True
                    stack.append(nb)
            if len(cell_ids) > best:
                best = len(cell_ids)
        return best


def run_one(params, seed, target_hours, wound_at_h=None, wound_defect=None,
            checkpoint_h=24):
    p = dict(params)
    if wound_defect is not None:
        p['defect'] = wound_defect
    sim = Sim(seed, p)
    steps_target = int(round(target_hours * 60 / K['dtMin']))
    wound_step = None
    if wound_at_h is not None:
        wound_step = int(round(wound_at_h * 60 / K['dtMin']))

    rows = []
    last_c = sim.count()
    while sim.step < steps_target:
        if wound_step is not None and sim.step == wound_step:
            sim.inflict_wound(False)
        last_c = sim.advance()
        elapsed_h = sim.step * K['dtMin'] / 60
        if checkpoint_h and (sim.step % int(round(checkpoint_h * 60 / K['dtMin'])) == 0):
            rows.append(_row(sim, last_c, params, seed, wound_at_h is not None, elapsed_h))

    final_row = _row(sim, last_c, params, seed, wound_at_h is not None, sim.step * K['dtMin'] / 60, final=True)
    return rows, final_row


def _row(sim, c, params, seed, has_wound, elapsed_h, final=False):
    return dict(
        seed=seed, elapsed_h=elapsed_h, final=final,
        has_wound=has_wound,
        tau_d=params.get('lifeD'), cyc_h=params.get('cycH'),
        Twake=params.get('Twake'), sigma=params.get('sigma'),
        pRevert=params.get('pRevert'), revTolH=params.get('revTolH'),
        revStore=params.get('revStore'),
        density=c['dens'], reserve_margin=(
            (c['nd'] + c['nv']) / max(1, c['norm'])
        ) / max(1e-9, sim.need_resv()),
        mean_area=c['aMean'], mobilized=c['mob'],
        clone_init=sim.clone_init, clone_max=sim.clone_max, clone_final=sim.clone_final,
        rev_total=sim.rev_total, rev_lost=sim.rev_lost,
        rev_established=sim.rev_established,
        rev_est_step_h=(sim.rev_est_step * K['dtMin'] / 60) if sim.rev_est_step >= 0 else None,
        shed_total=sim.shed_total, starve_total=sim.starve_total, div_total=sim.div_total,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=str, default='6,9,12')
    ap.add_argument('--wound', type=str, default='yes,no')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--hours', type=float, default=2000)
    ap.add_argument('--wound-at', type=float, default=500)
    ap.add_argument('--defect', type=float, default=600)
    ap.add_argument('--checkpoint', type=float, default=100)
    ap.add_argument('--out', type=str, default='results.csv')
    args = ap.parse_args()

    taus = [float(x) for x in args.tau.split(',')]
    wounds = [x.strip().lower() in ('yes', 'y', 'true', '1') for x in args.wound.split(',')]

    t0 = time.time()
    all_rows = []
    combo_i = 0
    total_combos = len(taus) * len(wounds)
    for tau in taus:
        for has_wound in wounds:
            combo_i += 1
            for seed in range(args.n):
                params = dict(lifeD=tau)
                rows, final_row = run_one(
                    params, seed, args.hours,
                    wound_at_h=args.wound_at if has_wound else None,
                    wound_defect=args.defect,
                    checkpoint_h=args.checkpoint,
                )
                all_rows.extend(rows)
                all_rows.append(final_row)
                print(f"[{combo_i}/{total_combos}] tau={tau} wound={has_wound} "
                      f"seed={seed} done in {time.time()-t0:.1f}s total", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out} in {time.time()-t0:.1f}s")
