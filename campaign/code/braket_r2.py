#!/usr/bin/env python3
"""R2: Rydberg-chain AHS program for QuEra Aquila via AWS Braket, with a
free local-emulator mode for the full dress rehearsal.

Protocol per detuning point (standard quasi-adiabatic preparation):
  1. Omega ramps 0 -> Omega0 over t_ramp at large negative detuning
  2. detuning sweeps linearly to the target Delta over t_sweep
  3. Omega ramps back to 0 over t_ramp; site-resolved readout

The geometry knob R_b/a is set through the lattice spacing a, since
C6 = 5.42e-24 rad m^6/s is fixed for Aquila (70S Rydberg state) and
R_b = (C6/Omega0)^(1/6).

Output rows: [delta_over_omega, b_1..b_N] with b = occupation (1 = Rydberg),
i.e. exactly the format of rydberg_dataset.py / ryd_tokenize.py.  Shots with
imperfect loading (pre_sequence != 1 anywhere) are discarded.

Usage (emulator, free):
  python braket_r2.py --device local --natoms 11 --rb-over-a 1.2 \
      --deltas -1.0:4.0:6 --shots 100 --json-out r2_local_test.json
Usage (Aquila, paid — needs AWS credentials configured):
  python braket_r2.py --device aquila --natoms 100 --rb-over-a 1.2 \
      --deltas -1.0:4.0:21 --shots 1000 --json-out r2_aquila_rb1.2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C6 = 5.42e-24            # rad m^6 / s (Aquila)
OMEGA0 = 2 * np.pi * 1.0e6   # rad/s working Rabi frequency
T_RAMP = 0.5e-6
T_SWEEP = 3.0e-6
DELTA_INIT = -2 * np.pi * 3.0e6


def parse_grid(spec):
    lo, hi, n = spec.split(':')
    return np.linspace(float(lo), float(hi), int(n))


def build_program(natoms, spacing, delta_over_omega):
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    register = AtomArrangement()
    for i in range(natoms):
        register.add([i * spacing, 0.0])

    t_total = 2 * T_RAMP + T_SWEEP
    delta_final = float(delta_over_omega) * OMEGA0

    amplitude = TimeSeries()
    amplitude.put(0.0, 0.0).put(T_RAMP, OMEGA0) \
             .put(T_RAMP + T_SWEEP, OMEGA0).put(t_total, 0.0)
    detuning = TimeSeries()
    detuning.put(0.0, DELTA_INIT).put(T_RAMP, DELTA_INIT) \
            .put(T_RAMP + T_SWEEP, delta_final).put(t_total, delta_final)
    phase = TimeSeries()
    phase.put(0.0, 0.0).put(t_total, 0.0)

    drive = DrivingField(amplitude=amplitude, detuning=detuning, phase=phase)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def get_device(kind):
    if kind == 'local':
        from braket.devices import LocalSimulator
        return LocalSimulator('braket_ahs')
    from braket.aws import AwsDevice
    return AwsDevice('arn:aws:braket:us-east-1::device/qpu/quera/Aquila')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['local', 'aquila'], default='local')
    ap.add_argument('--natoms', type=int, default=11)
    ap.add_argument('--rb-over-a', type=float, default=1.2)
    ap.add_argument('--deltas', type=str, default='-1.0:4.0:6')
    ap.add_argument('--shots', type=int, default=100)
    ap.add_argument('--json-out', type=str, required=True)
    ap.add_argument('--summary-out', type=str, default='')
    args = ap.parse_args()

    rb = (C6 / OMEGA0) ** (1.0 / 6.0)
    spacing = rb / args.rb_over_a
    if spacing < 4e-6:
        raise SystemExit(f"spacing {spacing*1e6:.2f} um < 4 um Aquila minimum "
                         f"(reduce rb-over-a or lower OMEGA0)")
    print(f"R_b = {rb*1e6:.2f} um, spacing a = {spacing*1e6:.2f} um "
          f"(R_b/a = {args.rb_over_a}), N = {args.natoms}, "
          f"device = {args.device}", flush=True)

    device = get_device(args.device)
    deltas = parse_grid(args.deltas)
    rows, summary = [], []
    for d in deltas:
        prog = build_program(args.natoms, spacing, d)
        if args.device == 'aquila':
            task = device.run(prog, shots=args.shots)
            print(f"  submitted delta/Omega={d:.2f}: {task.id}", flush=True)
            result = task.result()          # blocks until the QPU window runs it
        else:
            result = device.run(prog, shots=args.shots).result()
        kept = 0
        dens_acc = np.zeros(args.natoms)
        for m in result.measurements:
            pre = np.asarray(m.pre_sequence, dtype=int)
            post = np.asarray(m.post_sequence, dtype=int)
            if pre.min() < 1:               # imperfect loading -> discard shot
                continue
            occ = 1 - post                  # post=0 means Rydberg (atom gone dark)
            rows.append([float(d)] + occ.astype(int).tolist())
            dens_acc += occ
            kept += 1
        dens = dens_acc / max(kept, 1)
        summary.append((float(d), kept, float(dens.mean())))
        print(f"  delta/Omega={d:+.2f}: kept {kept}/{args.shots} shots, "
              f"<n>={dens.mean():.4f}", flush=True)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.json_out, 'w'))
    print(f"saved {len(rows)} snapshots -> {args.json_out}")
    if args.summary_out:
        np.savez(args.summary_out,
                 deltas=np.array([s[0] for s in summary]),
                 kept=np.array([s[1] for s in summary]),
                 mean_density=np.array([s[2] for s in summary]))
        print(f"saved summary -> {args.summary_out}")


if __name__ == '__main__':
    main()
