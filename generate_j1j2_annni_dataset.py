"""
Generate classical-shadow training data for J1-J2 and ANNNI spin chains.

The output format is compatible with train_oseq.py:

    [param, P1, b1, P2, b2, ..., PN, bN]

where P is encoded as 2=X, 3=Y, 4=Z and b is encoded as 0/1 for
negative/positive measurement outcomes.

Examples:

    python generate_j1j2_annni_dataset.py j1j2 --num-qubits 10 --samples-per-param 800 \
        --params 0.0 0.25 0.4 0.5 0.6 0.75 1.0 --json-out data/j1j2_10q_train.json

    python generate_j1j2_annni_dataset.py annni --num-qubits 10 --samples-per-param 800 \
        --params 0.0 0.25 0.5 0.75 1.0 --scan kappa --h 0.6 \
        --json-out data/annni_kappa_10q_train.json

    python generate_j1j2_annni_dataset.py annni --num-qubits 10 --samples-per-param 800 \
        --params 0.0 0.25 0.4 0.5 0.6 0.75 0.9 1.0 --scan h --kappa 0.4 \
        --json-out data/annni_h_10q_train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

try:
    import numpy as np
    from scipy.sparse import csr_matrix, kron as sparse_kron
    from scipy.sparse.linalg import eigsh

    SCIPY_AVAILABLE = True
except ImportError:
    np = None
    csr_matrix = None
    sparse_kron = None
    eigsh = None
    SCIPY_AVAILABLE = False


DTYPE = torch.cfloat
REAL_DTYPE = torch.float64

I2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=DTYPE)
Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)

H_MEASURE = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=DTYPE) / (2**0.5)
_, Y_EVECS = torch.linalg.eigh(Y)
Y_MEASURE = Y_EVECS[:, [1, 0]].contiguous()

if SCIPY_AVAILABLE:
    SP_I2_REAL = csr_matrix(np.eye(2, dtype=np.float64))
    SP_X_REAL = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    SP_Z_REAL = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64))
    SP_Y_COMPLEX = csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128))


def kron_all(ops: Iterable[torch.Tensor]) -> torch.Tensor:
    result = None
    for op in ops:
        result = op if result is None else torch.kron(result, op)
    if result is None:
        raise ValueError("ops must be non-empty")
    return result


def one_site_operator(num_qubits: int, site: int, op: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(2, dtype=op.dtype, device=op.device)
    ops = [identity] * num_qubits
    ops[site % num_qubits] = op
    return kron_all(ops)


def two_site_operator(num_qubits: int, i: int, j: int, op_i: torch.Tensor, op_j: torch.Tensor) -> torch.Tensor:
    i %= num_qubits
    j %= num_qubits
    if i == j:
        raise ValueError("two-site operator needs two distinct sites")
    dtype = torch.promote_types(op_i.dtype, op_j.dtype)
    identity = torch.eye(2, dtype=dtype, device=op_i.device)
    ops = [identity] * num_qubits
    ops[i] = op_i
    ops[j] = op_j
    return kron_all(ops)


def kron_all_sparse(ops: Iterable["csr_matrix"]) -> "csr_matrix":
    result = None
    for op in ops:
        result = op if result is None else sparse_kron(result, op, format="csr")
    if result is None:
        raise ValueError("ops must be non-empty")
    return result


def one_site_operator_sparse(num_qubits: int, site: int, op: "csr_matrix") -> "csr_matrix":
    ops = [SP_I2_REAL] * num_qubits
    ops[site % num_qubits] = op
    return kron_all_sparse(ops)


def two_site_operator_sparse(
    num_qubits: int,
    i: int,
    j: int,
    op_i: "csr_matrix",
    op_j: "csr_matrix",
) -> "csr_matrix":
    i %= num_qubits
    j %= num_qubits
    if i == j:
        raise ValueError("two-site operator needs two distinct sites")
    identity = SP_I2_REAL.astype(np.result_type(op_i.dtype, op_j.dtype), copy=False)
    ops = [identity] * num_qubits
    ops[i] = op_i
    ops[j] = op_j
    return kron_all_sparse(ops)


def hamiltonian_j1j2_pbc(num_qubits: int, j2: float, j1: float = 1.0) -> torch.Tensor:
    """
    Frustrated spin-1/2 J1-J2 chain with periodic boundary conditions.

    Pauli convention, matching the existing notebook's Heisenberg code:

        H = J1 sum_i (XX + YY + ZZ)_{i,i+1}
          + J2 sum_i (XX + YY + ZZ)_{i,i+2}

    If you want the S_i dot S_j convention, multiply both J1 and J2 by 1/4.
    The phase structure versus J2/J1 is unchanged by that global convention.
    """
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=REAL_DTYPE)
    x_op = X.real.double()
    z_op = Z.real.double()
    for i in range(num_qubits):
        nn = (i + 1) % num_qubits
        nnn = (i + 2) % num_qubits

        yy_nn = two_site_operator(num_qubits, i, nn, Y, Y).real.double()
        yy_nnn = two_site_operator(num_qubits, i, nnn, Y, Y).real.double()

        ham += j1 * (
            two_site_operator(num_qubits, i, nn, x_op, x_op)
            + yy_nn
            + two_site_operator(num_qubits, i, nn, z_op, z_op)
        )
        ham += j2 * (
            two_site_operator(num_qubits, i, nnn, x_op, x_op)
            + yy_nnn
            + two_site_operator(num_qubits, i, nnn, z_op, z_op)
        )
    return ham


def hamiltonian_annni_pbc(num_qubits: int, kappa: float, h: float, j1: float = 1.0) -> torch.Tensor:
    """
    ANNNI chain with periodic boundary conditions.

    Convention:

        H = -J1 sum_i Z_i Z_{i+1}
            +J2 sum_i Z_i Z_{i+2}
            -h  sum_i X_i

    with J2 = kappa * J1. Positive kappa frustrates the ferromagnetic nearest
    neighbour Ising term.
    """
    j2 = kappa * j1
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=REAL_DTYPE)
    for i in range(num_qubits):
        ham -= j1 * two_site_operator(num_qubits, i, i + 1, Z.real.double(), Z.real.double())
        ham += j2 * two_site_operator(num_qubits, i, i + 2, Z.real.double(), Z.real.double())
        ham -= h * one_site_operator(num_qubits, i, X.real.double())
    return ham


def hamiltonian_j1j2_sparse_pbc(num_qubits: int, j2: float, j1: float = 1.0) -> "csr_matrix":
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for sparse eigensolver mode")
    dim = 2**num_qubits
    ham = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(num_qubits):
        nn = (i + 1) % num_qubits
        nnn = (i + 2) % num_qubits
        yy_nn = two_site_operator_sparse(num_qubits, i, nn, SP_Y_COMPLEX, SP_Y_COMPLEX).real
        yy_nnn = two_site_operator_sparse(num_qubits, i, nnn, SP_Y_COMPLEX, SP_Y_COMPLEX).real

        ham += j1 * (
            two_site_operator_sparse(num_qubits, i, nn, SP_X_REAL, SP_X_REAL)
            + yy_nn
            + two_site_operator_sparse(num_qubits, i, nn, SP_Z_REAL, SP_Z_REAL)
        )
        ham += j2 * (
            two_site_operator_sparse(num_qubits, i, nnn, SP_X_REAL, SP_X_REAL)
            + yy_nnn
            + two_site_operator_sparse(num_qubits, i, nnn, SP_Z_REAL, SP_Z_REAL)
        )
    return ham


def hamiltonian_annni_sparse_pbc(num_qubits: int, kappa: float, h: float, j1: float = 1.0) -> "csr_matrix":
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for sparse eigensolver mode")
    dim = 2**num_qubits
    j2 = kappa * j1
    ham = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(num_qubits):
        ham -= j1 * two_site_operator_sparse(num_qubits, i, i + 1, SP_Z_REAL, SP_Z_REAL)
        ham += j2 * two_site_operator_sparse(num_qubits, i, i + 2, SP_Z_REAL, SP_Z_REAL)
        ham -= h * one_site_operator_sparse(num_qubits, i, SP_X_REAL)
    return ham


def ground_state_dense(ham: torch.Tensor) -> torch.Tensor:
    ham = (ham + ham.mH) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(ham)
    del eigenvalues
    psi = eigenvectors[:, 0].contiguous()
    return psi / psi.norm()


def ground_state_sparse(ham: "csr_matrix") -> torch.Tensor:
    eigenvalues, eigenvectors = eigsh(ham, k=1, which="SA", tol=1e-10, maxiter=max(1000, ham.shape[0] * 20))
    del eigenvalues
    psi_np = eigenvectors[:, 0]
    psi_np = psi_np / np.linalg.norm(psi_np)
    return torch.from_numpy(psi_np).to(dtype=DTYPE).contiguous()


def generate_quantum_measure_output_pure(psi: torch.Tensor) -> torch.Tensor:
    """
    Sequentially sample one classical shadow from a pure state.

    Returns shape [1, 2N], encoded as [P1, b1, P2, b2, ...].
    """
    num_qubits_float = torch.log2(torch.tensor(float(psi.numel()), dtype=torch.float64)).item()
    num_qubits = int(round(num_qubits_float))
    if 2**num_qubits != psi.numel():
        raise ValueError("psi length must be a power of 2")

    tokens = []
    psi_work = psi.reshape(-1).to(dtype=DTYPE).clone()

    for _ in range(num_qubits):
        pauli_token = int(torch.randint(2, 5, (1,)).item())
        measure_basis = pauli_token - 2  # 0:X, 1:Y, 2:Z

        left = psi_work.numel() // 2
        matrix = psi_work.view(2, left)
        if measure_basis == 0:
            matrix = H_MEASURE @ matrix
        elif measure_basis == 1:
            matrix = Y_MEASURE.conj().T @ matrix

        positive_prob = (matrix[0].conj() * matrix[0]).real.sum().clamp(0.0, 1.0)
        outcome = torch.bernoulli(positive_prob).to(torch.long)

        branch = matrix[0] if int(outcome.item()) == 1 else matrix[1]
        psi_work = (branch / (branch.norm() + 1e-30)).contiguous()

        tokens.extend([pauli_token, int(outcome.item())])

    return torch.tensor(tokens, dtype=torch.float32).view(1, -1)


def generate_dataset_for_params(
    *,
    model: str,
    num_qubits: int,
    params: list[float],
    samples_per_param: int,
    j1: float,
    eig_method: str = "auto",
    annni_scan: str = "kappa",
    annni_h: float = 0.6,
    annni_kappa: float = 0.4,
) -> torch.Tensor:
    rows = []
    use_sparse = eig_method == "sparse" or (eig_method == "auto" and num_qubits >= 12 and SCIPY_AVAILABLE)
    if eig_method == "sparse" and not SCIPY_AVAILABLE:
        raise RuntimeError("Requested --eig-method sparse, but scipy is not available")

    for param in tqdm(params, desc=f"{model} params"):
        if model == "j1j2":
            if use_sparse:
                ham = hamiltonian_j1j2_sparse_pbc(num_qubits=num_qubits, j1=j1, j2=param * j1)
            else:
                ham = hamiltonian_j1j2_pbc(num_qubits=num_qubits, j1=j1, j2=param * j1)
            output_param = param  # alpha = J2 / J1
        elif model == "annni":
            if annni_scan == "kappa":
                kappa = param
                h = annni_h
                output_param = kappa
            elif annni_scan == "h":
                kappa = annni_kappa
                h = param
                output_param = h
            else:
                raise ValueError(f"Unknown ANNNI scan mode: {annni_scan}")
            if use_sparse:
                ham = hamiltonian_annni_sparse_pbc(num_qubits=num_qubits, j1=j1, kappa=kappa, h=h)
            else:
                ham = hamiltonian_annni_pbc(num_qubits=num_qubits, j1=j1, kappa=kappa, h=h)
        else:
            raise ValueError(f"Unknown model: {model}")

        psi = ground_state_sparse(ham) if use_sparse else ground_state_dense(ham)
        param_column = torch.tensor([[output_param]], dtype=torch.float32)

        for _ in range(samples_per_param):
            shadow = generate_quantum_measure_output_pure(psi)
            rows.append(torch.cat((param_column, shadow), dim=1).squeeze(0))

    return torch.stack(rows, dim=0)


def save_dataset(data: torch.Tensor, json_out: str | None, pt_out: str | None) -> None:
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data.tolist(), f)
        print(f"saved json: {path}")

    if pt_out:
        path = Path(pt_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        print(f"saved pt: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate J1-J2 / ANNNI classical-shadow datasets.")
    parser.add_argument("model", choices=["j1j2", "annni"])
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--samples-per-param", type=int, default=800)
    parser.add_argument("--params", type=float, nargs="+", required=True)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--json-out", type=str, required=True)
    parser.add_argument("--pt-out", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eig-method",
        choices=["auto", "dense", "sparse"],
        default="auto",
        help="Ground-state solver. auto uses sparse eigsh for N>=12 when scipy is available.",
    )

    parser.add_argument(
        "--scan",
        choices=["kappa", "h"],
        default="kappa",
        help="ANNNI only: which scalar parameter is written in the first data column.",
    )
    parser.add_argument("--h", type=float, default=0.6, help="ANNNI only: fixed transverse field when --scan kappa.")
    parser.add_argument("--kappa", type=float, default=0.4, help="ANNNI only: fixed frustration when --scan h.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = generate_dataset_for_params(
        model=args.model,
        num_qubits=args.num_qubits,
        params=args.params,
        samples_per_param=args.samples_per_param,
        j1=args.j1,
        eig_method=args.eig_method,
        annni_scan=args.scan,
        annni_h=args.h,
        annni_kappa=args.kappa,
    )

    expected_width = 1 + 2 * args.num_qubits
    if data.shape[1] != expected_width:
        raise RuntimeError(f"Expected width {expected_width}, got {data.shape[1]}")
    print(f"dataset shape: {tuple(data.shape)}")
    print(f"format: [param, P1, b1, ..., P{args.num_qubits}, b{args.num_qubits}]")
    print(f"Pauli token range: {data[:, 1::2].min().item():.0f}..{data[:, 1::2].max().item():.0f}")
    print(f"outcome token range: {data[:, 2::2].min().item():.0f}..{data[:, 2::2].max().item():.0f}")

    save_dataset(data, json_out=args.json_out, pt_out=args.pt_out or None)


if __name__ == "__main__":
    main()
