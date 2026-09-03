# Q-WAVE: SqDRIFT hardware sampling + CISD seed + VAE subspace expansion + PT2

Reference implementation accompanying:

> S. Halder, C. Patra, and R. Maitra, *"Quantum Wavefunction Augmentation via
> Variational Encoders"* (Q-WAVE).

`main_sqdrift_hardware_v2_pt2.py` is a single, self-contained script that runs
the full pipeline for the N2 (2.5x bond length, 2.75 A) result from the paper:

1. Builds SqDRIFT Krylov circuits, submits them to IBM Quantum hardware (or
   loads the cached run shipped in this repo), and collects sampled Slater
   determinants.
2. Diagonalizes the hardware pool, augments it with CISD determinants, and
   diagonalizes the combined basis (`dice_core`, a fixed-basis Davidson
   solver).
3. Iteratively expands that basis with a VAE (`vae_generator.py`) that
   proposes new determinants from the current wavefunction, re-diagonalizing
   and keeping the run at its best energy until convergence.
4. Runs the semistochastic Epstein-Nesbet PT2 correction of Sharma, Holmes,
   Jeanmairet, Alavi, and Umrigar (*J. Chem. Theory Comput.* 2017, 13,
   1595-1604) on the converged wavefunction.

Steps 2-4's supporting code (the Dice-integral registration, and the PT2
matrix-element/estimator routines) is inlined directly into
`main_sqdrift_hardware_v2_pt2.py`; the only other project files it imports
are `diagonalizer_dice_cpp.py`, `vae_generator.py`, and the `dice_core`
extension.

Run parameters (molecule, basis, SqDRIFT/Krylov settings, backend, PT2
thresholds) are hardcoded near the top of the script rather than read from a
config file — edit them in place to run a different system.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Build the `dice_core` extension (needs a C++17 compiler and Eigen headers,
pulled in as a submodule):

```bash
git submodule add https://gitlab.com/libeigen/eigen dice_core/external/eigen
git submodule update --init --recursive
cd dice_core && python setup.py build_ext --inplace && cd ..
python dice_core/tests_smoke.py   # optional: validate against PySCF/numpy
```

See `dice_core/README.md` for build details, provenance, and API. It is
extracted from the **Dice** SHCI code (GPLv3) — see [License](#license).

### IBM Quantum credentials

The token is **not** hardcoded. Either export it before running:

```bash
export IBM_QUANTUM_TOKEN="..."
export IBM_QUANTUM_INSTANCE="crn:v1:bluemix:public:quantum-computing:..."
```

or save an account once beforehand with
`QiskitRuntimeService.save_account(...)` (see the
[Qiskit IBM Runtime docs](https://docs.quantum.ibm.com/guides/setup-channel));
the script picks up whichever account is already saved locally if the
environment variables are unset.

In practice this repo ships the actual hardware sample cache used for the
paper's N2 (2.5x) result
(`sample_data_nitrogen/nitrogen_2.5x_6-31g_sqdrift_krylov5_Nr100_N15_t0.15_shots5120.json`),
so running the script as-is reproduces that result directly from cache —
no IBM Quantum credentials needed unless you delete the cache file or change
the run parameters.

## Run

```bash
python main_sqdrift_hardware_v2_pt2.py
```

Per-iteration history (energy, basis dimension, PT2 result, etc.) is written
to `production_runs_nitrogen/<molecule>_hardware_history.json`, and a PT2
checkpoint to `production_runs_nitrogen/checkpoints/<molecule>_hardware.npz`.

## Repository layout

| File | Role |
|---|---|
| `main_sqdrift_hardware_v2_pt2.py` | Entry point: full pipeline, including the inlined PT2 correction |
| `diagonalizer_dice_cpp.py` | Registers integrals with, and calls, `dice_core` |
| `vae_generator.py` | β-annealed VAE that proposes new determinants |
| `dice_core/` | Fixed-basis CI diagonalizer (C++/pybind11, extracted from Dice) |
| `sample_data_nitrogen/` | Cached hardware sample counts for the N2 (2.5x) run |

## License

GPLv3 (see `LICENSE`) — required because `dice_core/` is GPLv3 code extracted
from the [Dice SHCI project](https://github.com/caleb-johnson/Dice) (Copyright
(c) 2017, Sandeep Sharma, with contributions from James E. T. Smith and Adam
A. Holmes), which this pipeline compiles and imports directly. See
`dice_core/README.md` for full provenance. Eigen, pulled in as a submodule
for building `dice_core`, is MPL2-licensed.
