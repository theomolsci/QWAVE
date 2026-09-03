"""Dice (SHCI) C++ fixed-basis diagonalizer, drop-in replacement for diagonalizer_dice.subspace_diagonalizer."""

import time
import numpy as np

_HCORE = None
_ERI_CHEM = None
_NORB = None

_DICE_CORE_HINT = (
    "dice_core extension not importable. Build it first:\n"
    "    cd dice_core && python setup.py build_ext --inplace"
)

_REGISTER_HINT = (
    "Integrals not registered. Add this one-line registration right after the "
    "qiskit-nature `hamiltonian` object is created in your main script:\n"
    "    import diagonalizer_dice_cpp\n"
    "    diagonalizer_dice_cpp.set_hamiltonian(hamiltonian)"
)


def _import_dice_core():
    try:
        import dice_core
    except ImportError as exc:
        raise ImportError(_DICE_CORE_HINT) from exc
    for fn in ("solve_fixed_basis", "diagonal_energies"):
        if not hasattr(dice_core, fn):
            raise ImportError(
                f"dice_core imported but has no '{fn}' - the extension is "
                f"not built yet (found {getattr(dice_core, '__file__', None)}).\n"
                + _DICE_CORE_HINT
            )
    return dice_core


def _popcount(arr_u64):
    """Population count of a uint64 array."""
    try:
        return np.bitwise_count(arr_u64)
    except AttributeError:
        return np.array([int(x).bit_count() for x in arr_u64], dtype=np.int64)


def _interleave_parity_signs(alpha_dets, beta_dets, norb):
    """Per-determinant sign (+/-1) relating blocked (JW) and interleaved (Dice) spin-orbital orderings."""
    alpha = np.ascontiguousarray(alpha_dets, dtype=np.uint64)
    beta = np.ascontiguousarray(beta_dets, dtype=np.uint64)
    swaps = np.zeros(alpha.shape, dtype=np.uint64)
    one = np.uint64(1)
    for k in range(norb - 1):
        b_k = (beta >> np.uint64(k)) & one
        n_a_above = _popcount(alpha >> np.uint64(k + 1)).astype(np.uint64)
        swaps += b_k * n_a_above
    return (1.0 - 2.0 * (swaps & one).astype(np.float64))


def set_hamiltonian(hamiltonian):
    """Register integrals from a qiskit-nature ElectronicEnergy (0.7.x)."""
    from qiskit_nature.second_q.operators.tensor_ordering import (
        find_index_order,
        to_chemist_ordering,
        IndexType,
    )
    from qiskit_nature.second_q.operators.symmetric_two_body import (
        SymmetricTwoBodyIntegrals,
        unfold,
    )

    hcore = np.asarray(hamiltonian.electronic_integrals.alpha["+-"],
                       dtype=np.float64)
    eri_tensor = hamiltonian.electronic_integrals.alpha["++--"]
    if isinstance(eri_tensor, SymmetricTwoBodyIntegrals):
        eri_raw = np.asarray(unfold(eri_tensor))
    else:
        eri_raw = np.asarray(eri_tensor)

    index_order = find_index_order(eri_raw)
    if index_order == IndexType.UNKNOWN:
        raise ValueError(
            "Could not determine the index ordering of the '++--' two-body "
            "tensor (IndexType.UNKNOWN). Convert it to chemist (ij|kl) order "
            "manually and call set_integrals(hcore, eri_chemist) instead."
        )
    eri_chem = np.asarray(to_chemist_ordering(eri_raw, index_order=index_order),
                          dtype=np.float64)
    print(f"    [INTEG] '++--' tensor detected as {index_order} "
          f"-> stored in CHEMIST (ij|kl) order")

    set_integrals(hcore, eri_chem)


def set_integrals(hcore, eri_chemist):
    """Register integrals directly: hcore (norb,norb), eri in CHEMIST order."""
    global _HCORE, _ERI_CHEM, _NORB
    hcore = np.ascontiguousarray(hcore, dtype=np.float64)
    eri_chemist = np.ascontiguousarray(eri_chemist, dtype=np.float64)
    if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
        raise ValueError(f"hcore must be square (norb,norb); got {hcore.shape}")
    norb = hcore.shape[0]
    if eri_chemist.shape != (norb,) * 4:
        raise ValueError(
            f"eri must have shape {(norb,)*4}; got {eri_chemist.shape}"
        )
    _HCORE, _ERI_CHEM, _NORB = hcore, eri_chemist, norb
    print(f"    [INTEG] Registered integrals for Dice C++ engine: "
          f"norb={norb} ({2*norb} qubits)")


def subspace_diagonalizer(bitstring_list, qubit_op,
                          n_roots=1, tol=1e-5, max_iter=100, max_subspace=40,
                          nuclear_repulsion=0.0, frozen_shift=0.0):
    """Fixed-basis ground-state diagonalization backed by the Dice SHCI C++ core."""
    if not bitstring_list:
        print("Error: empty bitstring list.")
        return 0.0, np.array([]), np.array([])

    if n_roots != 1:
        raise ValueError(
            f"diagonalizer_dice_cpp only supports n_roots=1 (got {n_roots})."
        )

    if _HCORE is None:
        raise RuntimeError(_REGISTER_HINT)

    dice_core = _import_dice_core()

    t0 = time.time()
    print("--- [INIT] Dice C++ Fixed-Basis Setup ---")

    n_qubits = len(bitstring_list[0].strip())
    norb = _NORB
    if n_qubits != 2 * norb:
        raise ValueError(
            f"bitstring length ({n_qubits} qubits) does not match the "
            f"registered integrals (norb={norb} -> {2*norb} qubits). "
            f"Did you register the right Hamiltonian via set_hamiltonian()?"
        )
    op_nq = getattr(qubit_op, "num_qubits", None)
    if op_nq is not None and op_nq != n_qubits:
        raise ValueError(
            f"qubit_op.num_qubits={op_nq} does not match bitstring length "
            f"{n_qubits}."
        )

    dets_sorted = sorted(int(str(s).strip(), 2) for s in bitstring_list)
    n_dets = len(dets_sorted)
    print(f"    [BASIS] {n_dets:,} determinants")

    alpha_mask = (1 << norb) - 1
    alpha_dets = np.array([d & alpha_mask for d in dets_sorted], dtype=np.uint64)
    beta_dets = np.array([d >> norb for d in dets_sorted], dtype=np.uint64)

    n_a = _popcount(alpha_dets)
    n_b = _popcount(beta_dets)
    sector_keys = np.stack([n_a, n_b], axis=1)
    unique_sectors, sector_inverse = np.unique(
        sector_keys, axis=0, return_inverse=True
    )
    n_sectors = len(unique_sectors)
    if n_sectors > 1:
        print(f"    [WARN] Determinants span {n_sectors} (n_alpha,n_beta) "
              f"sectors: {[tuple(s) for s in unique_sectors]} - solving only "
              f"the sector containing the lowest diagonal element "
              f"(matches diagonalizer_dice behavior).")

    global_min = np.inf
    min_sector = 0
    sector_indices = []
    for s in range(n_sectors):
        idx = np.nonzero(sector_inverse == s)[0]
        sector_indices.append(idx)
        diag_s = dice_core.diagonal_energies(
            np.ascontiguousarray(alpha_dets[idx]),
            np.ascontiguousarray(beta_dets[idx]),
            _HCORE, _ERI_CHEM,
        )
        m = float(np.min(diag_s))
        if m < global_min:
            global_min = m
            min_sector = s

    print(f"    [INIT]  Setup in {time.time() - t0:.1f}s")
    print(f"    [ITER 0] Ref E: "
          f"{global_min + nuclear_repulsion + frozen_shift:.6f}")

    idx = sector_indices[min_sector]
    t1 = time.time()
    print(f"\n--- [START] Dice C++ Davidson "
          f"(n_dets={len(idx):,}, sector=({int(unique_sectors[min_sector][0])},"
          f"{int(unique_sectors[min_sector][1])})) ---")
    elec_energy, ci_sector, n_iter = dice_core.solve_fixed_basis(
        np.ascontiguousarray(alpha_dets[idx]),
        np.ascontiguousarray(beta_dets[idx]),
        _HCORE, _ERI_CHEM,
        davidson_tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
    )
    print(f"    [DONE] Converged in {n_iter} iterations "
          f"(E_total={elec_energy + nuclear_repulsion + frozen_shift:.8f}, "
          f"t={time.time() - t1:.1f}s)")

    signs = _interleave_parity_signs(alpha_dets[idx], beta_dets[idx], norb)
    ci_vector = np.zeros(n_dets, dtype=np.float64)
    ci_vector[idx] = np.asarray(ci_sector, dtype=np.float64) * signs

    _dt = np.int64 if n_qubits < 64 else object
    _det_arr = np.array(dets_sorted, dtype=_dt)
    _shifts = np.arange(n_qubits - 1, -1, -1, dtype=_dt)
    bitstring_matrix = ((_det_arr[:, None] >> _shifts) & 1).astype(bool)

    print(f"Subspace dimension: {n_dets}")
    print(f"Lowest eigenvalue (Electronic): {elec_energy}")
    return elec_energy, ci_vector, bitstring_matrix
