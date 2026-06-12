"""Optimal candidate-budget allocation from empirical per-family gains.

THEORY.md §1 open task: measure per-family per-node gain distributions,
check win rates against the order-statistics prediction, and solve for the
optimal per-depth budget split {k_f} of the pool families.

Input: CSV from the engine's OQB_GAIN_LOG gate (node_seq,depth,ns,family,gain).
Families: x=axis best (per-node baseline, fixed backbone), a/b=inherited
mutations, c=cache blend, r=sparse random, s=root SIS-seeded, k=cache exact
(fixed backbone — every cached direction is always evaluated, not drawn).

Method (per depth bucket):
  1. Win-rate table: which family supplies the node's best pool gain.
  2. Marginal-value curve: for each family f and budget k, the bootstrap
     estimate of E[max of k draws from f] per node, normalized by the
     node's observed pool max (scale-free across nodes).
  3. Greedy allocation: the objective E[max] is concave in each k_f, so
     repeatedly granting +1 candidate to the family with the largest
     marginal increase yields the optimal {k_f} for a given total budget.

Drawn pool families only (a, b, c, r): axis and cache-exact are fixed
backbones, not part of the allocation question.

Usage: python research/family_allocation.py /tmp/gains_adult.csv [...]
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

DRAWN = ["a", "b", "c", "r"]
BUCKETS = [(0, 2), (3, 9)]  # shallow, deep
N_BOOT = 200
RNG = np.random.default_rng(0)


def load(path):
    """-> {node_id: (depth, ns, {fam: gains array})}"""
    nodes = {}
    with open(path) as f:
        for line in f:
            nid_s, d_s, ns_s, fam, g_s = line.rstrip("\n").split(",")
            nid = int(nid_s)
            if nid not in nodes:
                nodes[nid] = (int(d_s), int(ns_s), defaultdict(list))
            nodes[nid][2][fam].append(float(g_s))
    return nodes


def emax_boot(gains: np.ndarray, k: int) -> float:
    """Bootstrap E[max of k iid draws] from the empirical sample."""
    if len(gains) == 0 or k == 0:
        return 0.0
    draws = RNG.choice(gains, size=(N_BOOT, k), replace=True)
    return float(draws.max(axis=1).mean())


def analyze(path):
    nodes = load(path)
    print(f"\n=== {path}  ({len(nodes)} nodes) ===")

    for lo, hi in BUCKETS:
        sel = [v for v in nodes.values() if lo <= v[0] <= hi]
        if not sel:
            continue
        label = f"depth {lo}-{hi}"

        # 1. Win rates among ALL evaluated candidates (incl. backbones)
        wins = defaultdict(int)
        for _, _, fams in sel:
            best_f, best_g = None, 0.0
            for f, gs in fams.items():
                m = max(gs)
                if m > best_g:
                    best_g, best_f = m, f
            if best_f:
                wins[best_f] += 1
        tot = sum(wins.values()) or 1
        win_str = "  ".join(f"{f}:{100*wins[f]/tot:.1f}%"
                            for f in sorted(wins, key=lambda f: -wins[f]))
        print(f"\n[{label}]  {len(sel)} nodes   win-rate: {win_str}")

        # 2. Per-family marginal value curves (normalized per node)
        #    value(f, k) = mean over nodes of E[max k draws of f] / pool_max
        curves = {}
        for f in DRAWN:
            ks = [1, 2, 4, 8, 12, 16, 24, 32]
            vals = []
            for k in ks:
                per_node = []
                for _, _, fams in sel:
                    gs = np.asarray(fams.get(f, []))
                    pool = [g for ff in DRAWN for g in fams.get(ff, [])]
                    pm = max(pool) if pool else 0.0
                    if pm <= 0 or len(gs) < 3:
                        continue
                    per_node.append(emax_boot(gs, k) / pm)
                vals.append(np.mean(per_node) if per_node else 0.0)
            curves[f] = (ks, vals)
            print(f"  E[max k]/poolmax {f}: " +
                  "  ".join(f"k={k}:{v:.3f}" for k, v in zip(ks, vals)))

        # 3. Greedy optimal allocation for a total budget of 32
        #    (objective per node: E[max over union of family draws],
        #     approximated by max over per-family bootstrap maxima)
        budget = 32
        alloc = {f: 0 for f in DRAWN}
        # Pre-sample per-node per-family bootstrap max tables for speed
        node_tables = []
        for _, _, fams in sel[: min(len(sel), 800)]:
            pool = [g for ff in DRAWN for g in fams.get(ff, [])]
            pm = max(pool) if pool else 0.0
            if pm <= 0:
                continue
            tbl = {}
            for f in DRAWN:
                gs = np.asarray(fams.get(f, []))
                if len(gs) < 3:
                    tbl[f] = None
                    continue
                # cumulative max over 33 sequential draws × N_BOOT paths
                d = RNG.choice(gs, size=(N_BOOT, budget + 1), replace=True)
                tbl[f] = np.maximum.accumulate(d, axis=1) / pm
            node_tables.append(tbl)

        def obj(a):
            tot_v, n = 0.0, 0
            for tbl in node_tables:
                paths = None
                for f in DRAWN:
                    if a[f] == 0 or tbl[f] is None:
                        continue
                    fm = tbl[f][:, a[f] - 1]
                    paths = fm if paths is None else np.maximum(paths, fm)
                if paths is not None:
                    tot_v += paths.mean()
                    n += 1
            return tot_v / n if n else 0.0

        cur = 0.0
        for _ in range(budget):
            best_f, best_v = None, cur
            for f in DRAWN:
                a2 = dict(alloc); a2[f] += 1
                v = obj(a2)
                if v > best_v:
                    best_v, best_f = v, f
            if best_f is None:
                break
            alloc[best_f] += 1
            cur = best_v
        used = sum(alloc.values())
        print(f"  greedy optimal (budget 32): "
              + "  ".join(f"{f}:{alloc[f]}" for f in DRAWN)
              + f"   (used {used}, E[max]/poolmax = {cur:.3f})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
