# dice_core

Fixed-basis Slater-determinant CI diagonalizer, extracted from the **Dice**
SHCI code, exposed to Python as the `_dice_core` pybind11 extension.

## Provenance and license

The C++ in `src/` is extracted from the Dice semistochastic heat-bath CI
code, fork **caleb-johnson/Dice** (https://github.com/caleb-johnson/Dice),
the source the `qiskit-addon-dice-solver` binary is built from.  Dice was
developed by Sandeep Sharma with contributions from James E. T. Smith and
Adam A. Holmes (Copyright (c) 2017, Sandeep Sharma) and is licensed under the
**GNU General Public License v3** (or later).  This extraction is therefore
also **GPLv3**; the per-file license headers are retained.  Eigen (vendored
from the Dice repo into `external/eigen/`) is MPL2-licensed.

## What this does (and does not do)

Diagonalizes a molecular Hamiltonian in a **fixed, explicit list** of
(alpha, beta) determinant pairs — exactly the n pairs you pass, in that
order:

- **No** heat-bath/HCI determinant-space expansion (everything related to
  epsilon1 screening / `SHCIgetdeterminants` was removed).
- **No** Cartesian-product basis: the k-th basis state is
  (`alpha_dets[k]`, `beta_dets[k]`).

What was kept from Dice (unchanged algorithms):

- `Determinants.{h,cpp}` — bit-packed `Determinant`/`HalfDet`, parity, and
  the Slater–Condon matrix elements `Energy`, `Hij`, `Hij_1Excite`,
  `Hij_2Excite`.
- `SHCImakeHamiltonian.{h,cpp}` — the fast connection-finding machinery
  (`PopulateHelperLists2`: groups determinants by alpha/beta half-strings
  instead of brute-force O(n^2)) and the sparse-H build
  (`MakeHfromSMHelpers2` logic, lower triangle: float64 value + int32 column
  per connection).
- `Hmult.h` — the half-stored symmetric sparse matvec.
- `Davidson.{h,cpp}` — Dice's Davidson algorithm (subspace iteration,
  Rayleigh–Ritz, diagonal preconditioning, restart at `nroots+3`).
- `integral.h` — `oneInt`/`twoInt` with Dice's exact storage layout
  (one-body: dense over spin orbitals; two-body: 8-fold packed over spatial
  orbitals, chemist notation, plus `Direct`/`Exchange` matrices), filled
  directly from numpy arrays instead of an FCIDUMP file.

What was removed/replaced:

- MPI and all rank striping (compiled with `-DSERIAL`, single process).
- boost entirely: `boost::interprocess` shared-memory segments became plain
  heap `std::vector`s; `boost::serialization` (only used for disk batching
  of H, not needed); `boost::format` prints became `printf`.
- RDMs, SOC, perturbation theory, time-reversal symmetry (`Trev` forced 0),
  lexical-order hashing, disk-batched `SparseHam`.
- OpenMP is not used; the two hot spots are parallelized with `std::thread`:
  (a) sparse-H construction (rows distributed by alpha string — each
  determinant owns exactly one slot, so the matrix is bit-for-bit independent
  of the thread count), and (b) the `Hmult2` matvec inside Davidson
  (per-thread accumulators reduced in fixed order, deterministic for a given
  `n_threads`).
- Davidson's hard-coded `800*nroots` iteration cap / `exit(0)` became a
  caller-supplied `max_iter` that returns the current best estimate with a
  warning, and fatal `exit(0)` calls became C++ exceptions.

## Conventions

- `alpha_dets[k]` / `beta_dets[k]` (`np.uint64`): bit i (LSB = orbital 0) is
  the occupation of alpha/beta **spatial** orbital i in determinant k.
  Internally Dice packs spin orbital `2i` = alpha_i, `2i+1` = beta_i.
- `hcore`: float64 `(norb, norb)`; `eri`: float64 `(norb,)*4` in **chemist**
  notation `(ij|kl)` (what `pyscf.tools.fcidump.from_integrals` receives in
  the qiskit-addon-dice-solver pipeline). `eri` is assumed 8-fold symmetric.
- Core energy is implicitly 0: returned energies are pure active-space
  electronic energies `<psi|H|psi>`.
- Up to `norb = 64` spatial orbitals (`DetLen = 2`, i.e. 128 spin orbitals).
- All pairs must be unique and share the same `(n_alpha, n_beta)`;
  violations raise `ValueError`.

## API (re-exported by `dice_core/__init__.py`)

```python
solve_fixed_basis(alpha_dets, beta_dets, hcore, eri,
                  davidson_tol=1e-5, max_iter=100, max_subspace=40,
                  n_threads=0, verbose=True)
    -> (energy: float, ci: np.ndarray (n,), n_iter: int)
    # Ground state (1 root). ci is in input determinant order.
    # max_subspace = Davidson maxCopies (clamped to >= 5 and <= n).
    # n_threads = 0 -> std::thread::hardware_concurrency().

diagonal_energies(alpha_dets, beta_dets, hcore, eri, n_threads=0)
    -> np.ndarray (n,)   # <D_k|H|D_k> via Determinant::Energy

build_dense_hamiltonian(alpha_dets, beta_dets, hcore, eri)
    -> np.ndarray (n, n)  # debug/test helper; refuses n > 4000
```

## Build

```bash
cd dice_core
/path/to/venv/bin/python setup.py build_ext --inplace
```

Requires pybind11 >= 2.10 and a C++17 compiler; Eigen is vendored. On macOS
Command Line Tools installs whose toolchain lacks the libc++ headers,
`setup.py` automatically adds `-stdlib++-isystem <SDK>/usr/include/c++/v1`.

## Tests

```bash
/path/to/venv/bin/python tests_smoke.py
```

Validates against `pyscf.fci.direct_spin1` and dense `numpy.linalg.eigh`
(full sectors, hand-checkable H2-like case, determinant subsets to prove no
expansion happens, thread-count determinism, input validation).
