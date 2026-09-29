#!/usr/bin/env python3
"""Is the delta = 5 advantage a commensurability artefact of N = 32?

A Z3 crystal needs a chain length divisible by 3.  N = 32 is not, so near the
commensurate filling an extra atom cannot sit anywhere in particular and
delocalises -- which is exactly the fluctuation the completion exploits.  If
that is the whole story, a chain with N = 33 should show no headroom at all.

For each N: DMRG ground state, 1000 sampled shots, random 30% atom loss, then
   trivial   = E[max(n_i, 1-n_i)] from the density profile alone
   exact     = E[max(p, 1-p)] from the state conditioned on the survivors
The gap between them is the information the survivors carry.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ryd_bayes_bound import ground_state, mps_tensors, conditional_marginals  # noqa: E402

DELTAS = (3.4, 5.0)
LENGTHS = (32, 33)
SHOTS, LOSS = 600, 0.3

print(f"{'N':>4} {'N%3':>4} {'delta':>6} {'<n>':>6} {'trivial %':>10} {'exact limit %':>14} {'headroom':>9}")
for n in LENGTHS:
    for dl in DELTAS:
        psi = ground_state(n, 2.35, dl, 128, 5)
        dens = np.asarray(psi.expectation_value('Sz')) + 0.5
        rng = np.random.default_rng(3)
        snaps = np.empty((SHOTS, n), dtype=float)
        for t in range(SHOTS):
            sig, _ = psi.sample_measurements(rng=rng)
            snaps[t] = 1.0 - np.asarray(sig, dtype=float)      # index 0 = occupied
        T = mps_tensors(psi, 64)
        k = round(LOSS * n)
        hid = np.zeros((SHOTS, n), dtype=bool)
        for t in range(SHOTS):
            hid[t, rng.choice(n, size=k, replace=False)] = True
        p = conditional_marginals(T, snaps, hid)
        ph = p[hid]
        site = np.tile(np.arange(n), (SHOTS, 1))[hid]
        triv = 100 * np.mean(np.maximum(dens[site], 1 - dens[site]))
        ceil = 100 * np.mean(np.maximum(ph, 1 - ph))
        print(f"{n:>4} {n % 3:>4} {dl:>6.1f} {dens.mean():>6.3f} {triv:>10.2f} {ceil:>14.2f} {ceil-triv:>+9.2f}",
              flush=True)
