"""dice_core: fixed-basis determinant CI diagonalizer extracted from Dice.

Extracted from the Dice SHCI code (https://github.com/caleb-johnson/Dice),
GPLv3.  See README.md for provenance and API details.
"""

try:
    from ._dice_core import (
        build_dense_hamiltonian,
        diagonal_energies,
        solve_fixed_basis,
    )
except ImportError:  # pragma: no cover - direct (non-package) usage
    from _dice_core import (
        build_dense_hamiltonian,
        diagonal_energies,
        solve_fixed_basis,
    )

__all__ = [
    "solve_fixed_basis",
    "diagonal_energies",
    "build_dense_hamiltonian",
]
