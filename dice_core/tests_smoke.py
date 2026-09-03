"""Smoke tests for dice_core against pyscf FCI and dense diagonalization.

Run with:
    <venv>/bin/python tests_smoke.py
from inside dice_core/ (or anywhere; paths are handled below).
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dice_core  # noqa: E402

from pyscf import ao2mo  # noqa: E402
from pyscf.fci import direct_spin1  # noqa: E402


def strings(norb, nelec):
    """All occupation bitstrings (uint64) of nelec electrons in norb orbitals."""
    out = []
    for occ in itertools.combinations(range(norb), nelec):
        s = 0
        for o in occ:
            s |= 1 << o
        out.append(s)
    return np.array(out, dtype=np.uint64)


def full_basis(norb, nalpha, nbeta):
    """Cartesian product of all alpha and beta strings (full sector)."""
    sa = strings(norb, nalpha)
    sb = strings(norb, nbeta)
    alpha = np.repeat(sa, len(sb))
    beta = np.tile(sb, len(sa))
    return alpha, beta


def random_integrals(norb, seed):
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    raw = rng.standard_normal((norb,) * 4)
    # restore full 8-fold symmetry via pyscf's packing round-trip
    eri = ao2mo.restore(1, ao2mo.restore(8, raw, norb), norb)
    return np.ascontiguousarray(hcore), np.ascontiguousarray(eri)


def check(label, val, ref, tol):
    diff = abs(val - ref)
    status = "ok" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {val:.12f} vs {ref:.12f} (|diff|={diff:.3e}, tol={tol:g})")
    assert diff <= tol, f"{label}: |{val} - {ref}| = {diff} > {tol}"


def test_h2_like():
    print("== Test 1: norb=2, nelec=(1,1) (H2-like), hand-checkable ==")
    norb = 2
    hcore = np.array([[-1.25, -0.50], [-0.50, -0.48]])
    eri = np.zeros((2, 2, 2, 2))
    # chemist notation (ij|kl), 8-fold symmetric assignments
    pairs = {
        (0, 0, 0, 0): 0.675,
        (1, 1, 1, 1): 0.625,
        (0, 0, 1, 1): 0.660,
        (0, 1, 0, 1): 0.180,
        (0, 0, 0, 1): 0.120,
        (0, 1, 1, 1): 0.080,
    }
    for (i, j, k, l), v in pairs.items():
        for a, b in ((i, j), (j, i)):
            for c, d in ((k, l), (l, k)):
                eri[a, b, c, d] = v
                eri[c, d, a, b] = v

    alpha, beta = full_basis(norb, 1, 1)  # 4 determinants

    e_pyscf = direct_spin1.kernel(hcore, eri, norb, (1, 1), ecore=0.0)[0]

    H = dice_core.build_dense_hamiltonian(alpha, beta, hcore, eri)
    assert np.allclose(H, H.T, atol=1e-14)
    e_dense = np.linalg.eigh(H)[0][0]
    check("eigh(dense H) vs pyscf FCI", e_dense, e_pyscf, 1e-10)

    e, ci, nit = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-8, verbose=False
    )
    check("solve_fixed_basis vs pyscf FCI", e, e_pyscf, 1e-8)
    assert ci.shape == (len(alpha),)
    assert abs(np.linalg.norm(ci) - 1.0) < 1e-8

    d = dice_core.diagonal_energies(alpha, beta, hcore, eri)
    assert np.allclose(d, np.diag(H), atol=1e-12), "diagonal_energies mismatch"
    print("  [ok] diagonal_energies matches dense diagonal")


def test_random_full_sector():
    print("== Test 2: norb=5, nelec=(2,2), random integrals, FULL sector ==")
    norb = 5
    hcore, eri = random_integrals(norb, seed=12345)
    alpha, beta = full_basis(norb, 2, 2)  # 10*10 = 100 determinants
    print(f"  n_dets = {len(alpha)}")

    e_pyscf = direct_spin1.kernel(hcore, eri, norb, (2, 2), ecore=0.0)[0]

    H = dice_core.build_dense_hamiltonian(alpha, beta, hcore, eri)
    e_dense = np.linalg.eigh(H)[0][0]

    e, ci, nit = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-7, max_iter=200, verbose=False
    )
    check("solve_fixed_basis vs pyscf FCI", e, e_pyscf, 1e-8)
    check("solve_fixed_basis vs eigh(dense H)", e, e_dense, 1e-8)

    d = dice_core.diagonal_energies(alpha, beta, hcore, eri)
    assert np.allclose(d, np.diag(H), atol=1e-12), "diagonal_energies mismatch"
    print("  [ok] diagonal_energies matches dense diagonal")

    # CI vector check: residual of the dense eigenproblem
    res = np.linalg.norm(H @ ci - e * ci)
    print(f"  [ok] ||H ci - E ci|| = {res:.3e}")
    assert res < 1e-5

    return hcore, eri, alpha, beta


def test_subset(hcore, eri, alpha, beta):
    print("== Test 3: random SUBSET of determinants (proves no expansion) ==")
    rng = np.random.default_rng(777)
    idx = np.sort(rng.choice(len(alpha), size=37, replace=False))
    sa, sb = alpha[idx], beta[idx]

    H = dice_core.build_dense_hamiltonian(sa, sb, hcore, eri)
    e_dense = np.linalg.eigh(H)[0][0]

    e, ci, nit = dice_core.solve_fixed_basis(
        sa, sb, hcore, eri, davidson_tol=1e-7, max_iter=200, verbose=False
    )
    check("solve_fixed_basis (subset) vs eigh(dense H on subset)", e, e_dense, 1e-8)

    e_pyscf = direct_spin1.kernel(hcore, eri, 5, (2, 2), ecore=0.0)[0]
    assert e > e_pyscf + 1e-6, (
        "subset energy should lie strictly above full-sector FCI "
        "(variational, and proves the space was not expanded)"
    )
    print(f"  [ok] subset energy {e:.10f} > full FCI {e_pyscf:.10f} (variational)")


def test_threads(hcore, eri, alpha, beta):
    print("== Test 4: n_threads=1 vs n_threads=4 ==")
    e1, ci1, _ = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-7, n_threads=1, verbose=False
    )
    e4, ci4, _ = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-7, n_threads=4, verbose=False
    )
    check("n_threads=1 vs n_threads=4 energy", e1, e4, 1e-10)
    d1 = dice_core.diagonal_energies(alpha, beta, hcore, eri, n_threads=1)
    d4 = dice_core.diagonal_energies(alpha, beta, hcore, eri, n_threads=4)
    assert np.array_equal(d1, d4)
    print("  [ok] diagonal_energies identical across thread counts")


def test_validation():
    print("== Test 5: input validation ==")
    h = np.eye(2)
    eri = np.zeros((2, 2, 2, 2))
    a = np.array([1, 2], dtype=np.uint64)
    b = np.array([1, 1], dtype=np.uint64)

    # mismatched (n_alpha, n_beta)
    bad_b = np.array([1, 3], dtype=np.uint64)
    for args, msg in [
        ((a, bad_b, h, eri), "inconsistent (n_alpha,n_beta)"),
        ((np.array([1, 1], dtype=np.uint64), b, h, eri), "duplicate pairs"),
        ((np.array([1, 4], dtype=np.uint64), b, h, eri), "orbital beyond norb"),
    ]:
        try:
            dice_core.solve_fixed_basis(*args, verbose=False)
        except ValueError as exc:
            print(f"  [ok] ValueError for {msg}: {exc}")
        else:
            raise AssertionError(f"expected ValueError for {msg}")

    # dense builder size refusal (valid dets, n = 4070 > 4000)
    sa = strings(20, 3)  # 1140 strings, all popcount 3
    big_a = np.repeat(sa[:55], 74)
    big_b = np.tile(sa[:74], 55)
    try:
        dice_core.build_dense_hamiltonian(
            big_a, big_b, np.eye(20), np.zeros((20,) * 4)
        )
    except ValueError as exc:
        print(f"  [ok] ValueError for n > 4000: {exc}")
    else:
        raise AssertionError("expected ValueError for n > 4000")


def test_larger_sector():
    print("== Test 6: norb=8, nelec=(4,4), full sector (4900 dets) ==")
    norb = 8
    hcore, eri = random_integrals(norb, seed=2024)
    alpha, beta = full_basis(norb, 4, 4)
    print(f"  n_dets = {len(alpha)}")

    e_pyscf = direct_spin1.kernel(hcore, eri, norb, (4, 4), ecore=0.0)[0]

    import time

    t0 = time.time()
    e1, ci1, nit1 = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-7, max_iter=200,
        n_threads=1, verbose=False,
    )
    t1 = time.time() - t0
    t0 = time.time()
    e4, ci4, nit4 = dice_core.solve_fixed_basis(
        alpha, beta, hcore, eri, davidson_tol=1e-7, max_iter=200,
        n_threads=4, verbose=False,
    )
    t4 = time.time() - t0
    check("solve_fixed_basis vs pyscf FCI", e1, e_pyscf, 1e-8)
    check("n_threads=1 vs n_threads=4 energy", e1, e4, 1e-10)
    print(f"  [ok] niter={nit1}; t(1 thread)={t1:.2f}s t(4 threads)={t4:.2f}s")


if __name__ == "__main__":
    test_h2_like()
    hcore, eri, alpha, beta = test_random_full_sector()
    test_subset(hcore, eri, alpha, beta)
    test_threads(hcore, eri, alpha, beta)
    test_validation()
    test_larger_sector()
    print("\nALL SMOKE TESTS PASSED")
