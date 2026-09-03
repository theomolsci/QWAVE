"""SqDRIFT pipeline: hardware sampling + CISD seed + VAE-ML loop + semistochastic PT2."""

import gc
import json
import os
import warnings
import psutil, os as _os

def _ram(label=""):
    proc = psutil.Process(_os.getpid())
    rss  = proc.memory_info().rss / (1024 ** 3)
    avail = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  [RAM] {label:<45}  used={rss:.2f} GB  avail={avail:.2f} GB")

import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from vae_generator import generate_new_determinants

warnings.filterwarnings("ignore")

QiskitRuntimeService.save_account(
        overwrite=True,
        token = os.environ["IBM_QUANTUM_TOKEN"],
        instance=os.environ["IBM_QUANTUM_INSTANCE"])


script_dir = os.path.dirname(os.path.abspath(__file__))

# N2, 2.75 A (2.5x equilibrium bond length)
mol_name     = "nitrogen_2.5x"
mol_geometry = "N 0.0 0.0 0.0; N 0.0 0.0 2.75"
mol_basis    = "6-31g"
mol_charge   = 0
mol_spin     = 0
mol_units    = "angstrom"
freeze_core  = True

shots               = 5120
optimization_level  = 3
use_least_busy      = False
backend_name        = "ibm_kingston"

Nr          = 100
N           = 15
t           = 0.15
ext_sqdrift = False

krylov_dim = 5
if krylov_dim % 2 == 0:
    print(f"WARNING: krylov_dim={krylov_dim} is even; the SqDRIFT paper requires an odd Krylov dimension.")

VAE_LATENT_DIM = 180
VAE_N_PRIOR    = 500000

cache_filename = (
    f"{mol_name}_{mol_basis}_sqdrift_krylov{krylov_dim}_"
    f"Nr{Nr}_N{N}_t{t}_shots{shots}.json"
)
sample_dir = os.path.join(script_dir, "sample_data_nitrogen")
output_dir = os.path.join(script_dir, "production_runs_nitrogen")
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
cache_path = os.path.join(sample_dir, cache_filename)

unit_map = {"bohr": DistanceUnit.BOHR, "angstrom": DistanceUnit.ANGSTROM}
distance_unit = unit_map.get(mol_units, DistanceUnit.ANGSTROM)

driver = PySCFDriver(
    atom=mol_geometry,
    basis=mol_basis,
    charge=mol_charge,
    spin=mol_spin,
    unit=distance_unit,
)
problem = driver.run()

if freeze_core:
    problem = FreezeCoreTransformer().transform(problem)

mapper      = JordanWignerMapper()
hamiltonian = problem.hamiltonian
second_q_op = hamiltonian.second_q_op()

qubit_op        = mapper.map(second_q_op)

import diagonalizer_dice_cpp
diagonalizer_dice_cpp.set_hamiltonian(hamiltonian)

operator_labels, coefficients = zip(*second_q_op.items())
operator_labels = list(operator_labels)
coefficients    = list(coefficients)

nuclear_repulsion_energy = hamiltonian.nuclear_repulsion_energy
num_particles            = problem.num_particles
num_spatial_orbitals     = problem.num_spatial_orbitals

if freeze_core:
    fc_shift = hamiltonian.constants['FreezeCoreTransformer']
else:
    fc_shift = 0.0

n_qubits = 2 * num_spatial_orbitals

_ram("after imports + driver.run()")
print("===============")
print("Basis:                    ", problem.basis)
print("Num Particles:            ", num_particles)
print("Num Spatial Orbitals:     ", num_spatial_orbitals)
print("Nuclear Repulsion Energy: ", nuclear_repulsion_energy)
print("Nr:", Nr, " N:", N, " t:", t, " Ext-SqDRIFT:", ext_sqdrift)

COEFF_THRESH = 1e-12

def conjugate_label(label: str) -> str:
    tokens = label.strip().split()
    swapped = []
    for tok in tokens:
        if tok.startswith("+_"):
            swapped.append("-_" + tok[2:])
        else:
            swapped.append("+_" + tok[2:])
    return " ".join(reversed(swapped))

seen_labels     = set()
terms_filtered  = []
filtered_coeffs = []
term_signs      = []

for label, coeff in zip(operator_labels, coefficients):
    if label.strip() == "" or abs(coeff) < COEFF_THRESH:
        continue
    if label in seen_labels:
        continue
    conj = conjugate_label(label)
    seen_labels.add(label)
    seen_labels.add(conj)
    combined_coeff = abs(coeff)
    if abs(combined_coeff) < COEFF_THRESH:
        continue
    terms_filtered.append((label, conj))
    filtered_coeffs.append(combined_coeff)
    term_signs.append(float(np.sign(np.real(coeff))))

abs_coeffs = np.array(filtered_coeffs)
lam        = abs_coeffs.sum()
prob_dist  = abs_coeffs / lam

_ram("after qubit_op / mapper.map()")
print(f"\nTerms before filtering: {len(operator_labels)}")
print(f"Terms after filtering:  {len(terms_filtered)}")
print(f"lambda (one-norm):      {lam:.6f}")

def parse_fermionic_label(label: str) -> dict:
    creators, annihilators = [], []
    for token in label.strip().split():
        if token.startswith("+_"):
            creators.append(int(token[2:]))
        elif token.startswith("-_"):
            annihilators.append(int(token[2:]))
        else:
            raise ValueError(f"Unrecognised token '{token}' in label '{label}'")
    if len(creators) != len(annihilators):
        raise ValueError(f"Particle number not conserved in label '{label}'")
    return {"creators": creators, "annihilators": annihilators, "n_body": len(creators)}

parsed_terms = [parse_fermionic_label(lbl) for lbl, _ in terms_filtered]

n_one_body = sum(1 for p in parsed_terms if p["n_body"] == 1)
n_two_body = sum(1 for p in parsed_terms if p["n_body"] == 2)
print(f"\n1-body terms: {n_one_body}")
print(f"2-body terms: {n_two_body}")

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_nature.second_q.operators import FermionicOp


def f2q_optimize(sampled_indices, parsed_terms, n_qubits, n_iter=200):
    perm = np.arange(n_qubits)
    best_perm = perm.copy()

    mode_data = []
    for idx in sampled_indices:
        term = parsed_terms[idx]
        modes = list(dict.fromkeys(term["creators"] + term["annihilators"]))
        if len(modes) >= 2:
            mode_data.append((modes, term["n_body"]))

    def compute_cost(p):
        cost = 0
        for modes, n_body in mode_data:
            mapped = [p[m] for m in modes]
            if n_body == 1:
                cost += abs(mapped[0] - mapped[1])
            else:
                for a in range(len(mapped)):
                    for b in range(a + 1, len(mapped)):
                        cost += abs(mapped[a] - mapped[b])
        return cost

    pairs_a = np.random.randint(0, n_qubits, size=n_iter)
    pairs_b = (pairs_a + np.random.randint(1, n_qubits, size=n_iter)) % n_qubits

    best_cost = compute_cost(perm)
    for i, j in zip(pairs_a.tolist(), pairs_b.tolist()):
        perm[i], perm[j] = perm[j], perm[i]
        cost = compute_cost(perm)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm.copy()
        else:
            perm[i], perm[j] = perm[j], perm[i]
    return best_perm


def jw_map_term(term_pair, sign, perm, n_qubits, mapper):
    def permute_label(label):
        tokens = label.strip().split()
        permuted = []
        for tok in tokens:
            prefix, idx = tok[0], int(tok[2:])
            permuted.append(f"{prefix}_{perm[idx]}")
        return " ".join(permuted)

    label, conj = term_pair
    perm_label      = permute_label(label)
    perm_conj_label = permute_label(conj)
    ferm_op = FermionicOp({perm_label: sign, perm_conj_label: sign}, num_spin_orbitals=n_qubits)
    return mapper.map(ferm_op)


def permuted_hf_circuit(num_particles, num_spatial_orbitals, perm):
    n_alpha, n_beta = num_particles
    occupied_modes  = list(range(n_alpha)) + list(range(num_spatial_orbitals, num_spatial_orbitals + n_beta))
    from qiskit.circuit import QuantumCircuit
    circ = QuantumCircuit(len(perm))
    for m in occupied_modes:
        circ.x(perm[m])
    return circ


def build_qdrift_circuit(k, sampled_indices, perm, terms_filtered, parsed_terms, term_signs,
                         lam, t, N, num_particles, num_spatial_orbitals, n_qubits, mapper):
    circ  = permuted_hf_circuit(num_particles, num_spatial_orbitals, perm)
    angle = k * t * lam / N
    all_qargs = list(range(n_qubits))
    for j in sampled_indices:
        pauli_op = jw_map_term(terms_filtered[j], term_signs[j], perm, n_qubits, mapper)
        gate     = PauliEvolutionGate(pauli_op, time=angle, synthesis=LieTrotter(reps=1))
        circ.append(gate, qargs=all_qargs)
    circ.measure_all()
    return circ


def unpermute_bitstring(bitstring, perm):
    bits_q = [int(bitstring[-(q + 1)]) for q in range(len(perm))]
    original_bits = [bits_q[perm[m]] for m in range(len(perm))]
    return ''.join(str(b) for b in original_bits[::-1])


circuit_list  = []
circuit_perms = []

for k in range(krylov_dim):
    if k == 0:
        circ = permuted_hf_circuit(num_particles, num_spatial_orbitals, np.arange(n_qubits))
        circ.measure_all()
        circuit_list.append(circ)
        circuit_perms.append(np.arange(n_qubits).tolist())
    else:
        for r in range(Nr):
            sampled_indices = np.random.choice(len(prob_dist), size=N, replace=True, p=prob_dist)
            perm = f2q_optimize(sampled_indices, parsed_terms, n_qubits)
            circ = build_qdrift_circuit(k, sampled_indices, perm, terms_filtered, parsed_terms,
                                        term_signs, lam, t, N, num_particles, num_spatial_orbitals,
                                        n_qubits, mapper)
            circuit_list.append(circ)
            circuit_perms.append(perm.tolist())

print(f"\nTotal circuits built: {len(circuit_list)}")

if os.path.exists(cache_path):
    print(f"\nCache file found: {cache_path}")
    print("Loading counts from cache - skipping hardware run.")
    with open(cache_path, "r") as fh:
        cache_data = json.load(fh)
    all_counts    = cache_data["circuits"]
    circuit_perms = cache_data["perms"]

else:
    print(f"\nNo cache file found at {cache_path}.")
    print("Transpiling and submitting circuits to IBM Quantum hardware...")

    service = QiskitRuntimeService()

    if backend_name:
        backend = service.backend(backend_name)
    elif use_least_busy:
        backend = service.least_busy(operational=True, simulator=False)
    else:
        raise ValueError(
            "backend.name is null and backend.use_least_busy is false - "
            "cannot determine which backend to use."
        )

    print(f"Selected backend: {backend}")

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    isa_circuits = pm.run(circuit_list)

    two_qubit_gate_names = {"cx", "ecr", "cz", "rzz", "rxx", "ryy", "swap"}

    def circuit_stats(circ):
        ops = circ.count_ops()
        total  = sum(ops.values())
        two_q  = sum(v for k, v in ops.items() if k in two_qubit_gate_names)
        depth  = circ.depth()
        return total, two_q, depth

    print("\n=== Transpiled gate counts per Krylov index ===")
    print(f"{'k':>3}  {'circuits':>8}  {'total gates':>12}  {'2Q gates':>9}  {'depth':>6}")
    print("-" * 46)

    idx = 0
    for k in range(krylov_dim):
        n_circs = 1 if k == 0 else Nr
        group   = isa_circuits[idx : idx + n_circs]
        idx    += n_circs

        stats  = [circuit_stats(c) for c in group]
        totals = [s[0] for s in stats]
        two_qs = [s[1] for s in stats]
        depths = [s[2] for s in stats]

        if n_circs == 1:
            print(f"{k:>3}  {n_circs:>8}  {totals[0]:>12}  {two_qs[0]:>9}  {depths[0]:>6}")
        else:
            print(f"{k:>3}  {n_circs:>8}  "
                  f"{min(totals):>5}-{max(totals):<5}  "
                  f"{min(two_qs):>4}-{max(two_qs):<4}  "
                  f"{min(depths):>3}-{max(depths)}")

    print()
    print("Gate counts above. Submitting to hardware automatically")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots

    job = sampler.run(isa_circuits)
    print(f"Job ID: {job.job_id()}")
    result = job.result()

    all_counts = [result[i].data.meas.get_counts() for i in range(len(circuit_list))]

    metadata = {
        "molecule": {"name": mol_name, "geometry": mol_geometry, "basis": mol_basis,
                     "charge": mol_charge, "spin": mol_spin, "units": mol_units,
                     "freeze_core": freeze_core},
        "krylov":   {"dimension": krylov_dim},
        "backend":  {"shots": shots, "optimization_level": optimization_level,
                     "use_least_busy": use_least_busy, "name": backend_name},
        "sqdrift":  {"Nr": Nr, "N": N, "t": t, "ext_sqdrift": ext_sqdrift},
    }
    cache_data = {
        "circuits": all_counts,
        "perms":    circuit_perms,
        "metadata": metadata,
    }
    with open(cache_path, "w") as fh:
        json.dump(cache_data, fh, indent=2)
    print(f"Counts saved to cache: {cache_path}")

total_electrons  = sum(num_particles)
n_alpha_filt, n_beta_filt = num_particles
bitstring_pool   = set()
for counts, perm in zip(all_counts, circuit_perms):
    for bitstring in counts:
        unpermuted = unpermute_bitstring(bitstring, perm)
        if (unpermuted[num_spatial_orbitals:].count('1') == n_alpha_filt and
                unpermuted[:num_spatial_orbitals].count('1') == n_beta_filt):
            bitstring_pool.add(unpermuted)

total_raw = sum(sum(counts.values()) for counts in all_counts)
_ram("after hardware counts loaded")
print(f"Total raw bitstrings collected: {total_raw}")
print(f"Bitstrings after CR filtering:  {len(bitstring_pool)}")

from diagonalizer_dice_cpp import subspace_diagonalizer


def _fast_mirror_excitations(bitstring_list: list) -> list:
    """Vectorized mirror-excitation that swaps alpha/beta halves via integer ops."""
    if not bitstring_list:
        return []
    L = len(bitstring_list[0])
    if L % 2 != 0:
        raise ValueError("Bitstring length is odd.")
    half = L // 2

    arr = np.frompyfunc(int, 2, 1)(bitstring_list, 2).astype(object)

    mask_lower = (1 << half) - 1
    lower = arr & mask_lower
    upper = (arr >> half) & mask_lower
    mirrors = (lower << half) | upper

    fmt = f"{{:0{L}b}}"
    orig_strs    = np.array(bitstring_list, dtype=object)
    mirror_strs  = np.frompyfunc(fmt.format, 1, 1)(mirrors)

    combined = np.concatenate([orig_strs, mirror_strs])
    unique   = np.unique(combined)
    return unique.tolist()


bitstring_list = list(bitstring_pool)
print("Dimension of Diagonalisation", len(bitstring_list))
_hw_diag_dim = len(bitstring_list)
_ram("before HW diagonalization")
lowest_eigenvalue, eigenvector, bitstring_matrix = subspace_diagonalizer(bitstring_list, qubit_op)
_ram("after HW diagonalization")
final_energy = lowest_eigenvalue + nuclear_repulsion_energy + fc_shift
print("FINAL ENERGY TOTAL", final_energy)

n_alpha, n_beta = num_particles
occupied_modes  = set(range(n_alpha)) | set(range(num_spatial_orbitals, num_spatial_orbitals + n_beta))
hf_row = np.array([1 if (n_qubits - 1 - j) in occupied_modes else 0 for j in range(n_qubits)])

N_ML_ITERS    = 100
PROB_THRESH   = 1e-10
SAMPLING_TEMP = 2.0

from itertools import combinations as _combinations

_hf_bitstring = ''.join(str(int(b)) for b in hf_row)

_hw_probs    = np.abs(eigenvector) ** 2
_hw_bm_int   = bitstring_matrix.astype(int)
_hw_not_hf   = ~np.all(_hw_bm_int == hf_row, axis=1)
_hw_mask     = (_hw_probs > PROB_THRESH) & _hw_not_hf
_hw_chars    = _hw_bm_int[_hw_mask].astype(np.uint8) + np.uint8(ord('0'))
_filt_hw_strs = [row.tobytes().decode('ascii') for row in _hw_chars]

print("\nCISD augmentation:")
print(f"  Hardware determinants above threshold:  {len(_filt_hw_strs)}")

_occ  = np.where(hf_row == 1)[0]
_virt = np.where(hf_row == 0)[0]

_occ_alpha  = [i for i in _occ  if i >= num_spatial_orbitals]
_occ_beta   = [i for i in _occ  if i <  num_spatial_orbitals]
_virt_alpha = [i for i in _virt if i >= num_spatial_orbitals]
_virt_beta  = [i for i in _virt if i <  num_spatial_orbitals]

_cisd_pool = {_hf_bitstring}

for i in _occ_alpha:
    for a in _virt_alpha:
        det = hf_row.copy(); det[i] = 0; det[a] = 1
        _cisd_pool.add(''.join(map(str, det)))
for i in _occ_beta:
    for a in _virt_beta:
        det = hf_row.copy(); det[i] = 0; det[a] = 1
        _cisd_pool.add(''.join(map(str, det)))

for i, j in _combinations(_occ_alpha, 2):
    for a, b in _combinations(_virt_alpha, 2):
        det = hf_row.copy(); det[i] = 0; det[j] = 0; det[a] = 1; det[b] = 1
        _cisd_pool.add(''.join(map(str, det)))
for i, j in _combinations(_occ_beta, 2):
    for a, b in _combinations(_virt_beta, 2):
        det = hf_row.copy(); det[i] = 0; det[j] = 0; det[a] = 1; det[b] = 1
        _cisd_pool.add(''.join(map(str, det)))
for i in _occ_alpha:
    for a in _virt_alpha:
        for j in _occ_beta:
            for b in _virt_beta:
                det = hf_row.copy(); det[i] = 0; det[a] = 1; det[j] = 0; det[b] = 1
                _cisd_pool.add(''.join(map(str, det)))

_n_sd = len(_cisd_pool) - 1
print(f"  CISD space: {_n_sd} SD determinants + HF")

_ram("before CISD diagonalization")
_eig_cisd, _ev_cisd, _bm_cisd = subspace_diagonalizer(list(_cisd_pool), qubit_op)
_ram("after CISD diagonalization")
_energy_cisd = _eig_cisd + nuclear_repulsion_energy + fc_shift
print(f"           CISD energy: {_energy_cisd:.10f} Ha")

_cisd_probs    = np.abs(_ev_cisd) ** 2
_cisd_bm_int   = _bm_cisd.astype(int)
_cisd_not_hf   = ~np.all(_cisd_bm_int == hf_row, axis=1)
_cisd_mask     = (_cisd_probs > PROB_THRESH) & _cisd_not_hf
_cisd_chars    = _cisd_bm_int[_cisd_mask].astype(np.uint8) + np.uint8(ord('0'))
_filt_cisd_strs = [row.tobytes().decode('ascii') for row in _cisd_chars]
print(f"           CISD determinants above threshold:  {len(_filt_cisd_strs)}")

_cisd_diag_dim = len(_cisd_pool)
_init_diag_dim = _hw_diag_dim if _hw_diag_dim >= _cisd_diag_dim else _cisd_diag_dim

_combined_set  = set(_filt_hw_strs) | set(_filt_cisd_strs) | {_hf_bitstring}
_combined_list = list(_combined_set)
print(f"  Combined basis (filtered HW + filtered CISD): {len(_combined_list)}")

_ram("before combined diagonalization")
_eig_comb, _ev_comb, _bm_comb = subspace_diagonalizer(_combined_list, qubit_op)
_ram("after combined diagonalization")
_energy_comb = _eig_comb + nuclear_repulsion_energy + fc_shift
print(f"           Energy hardware-only:        {final_energy:.10f} Ha")
print(f"           Energy combined (HW + CISD): {_energy_comb:.10f} Ha")

_comb_probs    = np.abs(_ev_comb) ** 2
_comb_bm_int   = _bm_comb.astype(int)
_comb_not_hf   = ~np.all(_comb_bm_int == hf_row, axis=1)
_comb_mask     = (_comb_probs > PROB_THRESH) & _comb_not_hf
_comb_chars    = _comb_bm_int[_comb_mask].astype(np.uint8) + np.uint8(ord('0'))
_filt_comb_strs = [row.tobytes().decode('ascii') for row in _comb_chars]
_init_n_above_rec = len(_filt_comb_strs)
print(f"           Final filtered determinants for ML:  {_init_n_above_rec}")
print("=" * 76)

lowest_eigenvalue = _eig_comb
eigenvector       = _ev_comb
bitstring_matrix  = _bm_comb
final_energy      = _energy_comb
bitstring_list    = _filt_comb_strs + [_hf_bitstring]

E_CONV      = 1e-5
CONV_WINDOW = 5
stagnation  = 0

best_energy           = final_energy
best_bitstring_list   = bitstring_list.copy()
best_eigenvector      = eigenvector.copy()
best_bitstring_matrix = bitstring_matrix.copy()

_vae_state = None

_existing_pool_set: set = set(bitstring_list)

history_path = os.path.join(output_dir, f"{mol_name}_hardware_history.json")

history = {
    "iteration":          [0],
    "energy":             [final_energy],
    "diag_dimension":     [_init_diag_dim],
    "n_above_threshold":  [_init_n_above_rec],
    "new_ml_dets":        [0],
}
with open(history_path, "w") as fh:
    json.dump(history, fh, indent=2)

for ml_iter in range(N_ML_ITERS):
    print(f"\n[iter {ml_iter + 1}/{N_ML_ITERS}] PROB_THRESH = {PROB_THRESH:.2e}")

    _ram("iter start")
    probabilities = np.abs(eigenvector) ** 2
    bm_int        = bitstring_matrix.astype(int)
    mask          = (probabilities > PROB_THRESH) & ~np.all(bm_int == hf_row, axis=1)
    filtered_prob             = probabilities[mask].tolist()
    filtered_bitstring_matrix = bm_int[mask]
    del bm_int, probabilities
    gc.collect()
    _ram("after threshold filter")
    print(f"Dimension after diagonalisation and putting threshold: {len(filtered_prob)}")

    n_samples = 100000
    raw_prob = np.array(filtered_prob) ** (1.0 / SAMPLING_TEMP)
    raw_sum  = raw_prob.sum()
    if raw_sum < 1e-300 or not np.isfinite(raw_sum):
        raise RuntimeError(
            f"[iter {ml_iter + 1}] Probability array collapsed to zero/NaN after "
            f"temperature scaling (T={SAMPLING_TEMP}). "
            f"Check PROB_THRESH - no determinants may be above threshold."
        )
    normalised_filtered_prob = raw_prob / raw_sum

    sampled_indices    = np.random.choice(len(normalised_filtered_prob), size=n_samples,
                                      replace=True, p=normalised_filtered_prob)
    sampled_bitstrings = filtered_bitstring_matrix[sampled_indices]

    _ram("before VAE")
    new_determinants, _vae_state = generate_new_determinants(
        sampled_bitstrings=sampled_bitstrings,
        existing_pool=_existing_pool_set,
        total_electrons=total_electrons,
        n_qubits=n_qubits,
        n_alpha=n_alpha,
        n_beta=n_beta,
        num_spatial_orbitals=num_spatial_orbitals,
        latent_dim=VAE_LATENT_DIM,
        n_prior_samples=VAE_N_PRIOR,
        init_state_dict=_vae_state,
    )
    del sampled_bitstrings, sampled_indices, raw_prob, normalised_filtered_prob
    gc.collect()
    _ram("after VAE + gc")

    _chars = filtered_bitstring_matrix.astype(np.uint8) + np.uint8(ord('0'))
    carry_over_strs = [row.tobytes().decode('ascii') for row in _chars]
    del filtered_bitstring_matrix, filtered_prob, _chars
    gc.collect()

    expanded_list = _fast_mirror_excitations(carry_over_strs + new_determinants + [_hf_bitstring])
    del carry_over_strs
    _ram("after add_mirror_excitations")
    print(f"Expanded basis: {len(expanded_list)} bitstrings "
          f"({len(new_determinants)} new from ML model + HF state added back)")

    gc.collect()
    _ram("before ML diagonalization")
    lowest_eigenvalue_exp, eigenvector, bitstring_matrix = subspace_diagonalizer(expanded_list, qubit_op)
    _ram("after ML diagonalization")
    final_energy_exp = lowest_eigenvalue_exp + nuclear_repulsion_energy + fc_shift
    print(f"[iter {ml_iter + 1}] FINAL ENERGY (ML-expanded basis): {final_energy_exp}")

    if final_energy_exp < best_energy:
        delta_E               = abs(final_energy_exp - best_energy)
        best_energy           = final_energy_exp
        best_bitstring_list   = expanded_list.copy()
        best_eigenvector      = eigenvector.copy()
        best_bitstring_matrix = bitstring_matrix.copy()
        bitstring_list        = expanded_list.copy()
        _existing_pool_set.update(expanded_list)
        print(f"[iter {ml_iter + 1}] Energy improved by {delta_E:.2e} Ha - accepting new basis.")
        stagnation = stagnation + 1 if delta_E < E_CONV else 0
    else:
        eigenvector      = best_eigenvector.copy()
        bitstring_matrix = best_bitstring_matrix.copy()
        bitstring_list   = best_bitstring_list.copy()
        stagnation      += 1
        print(f"[iter {ml_iter + 1}] Energy did not improve - rolling back to best basis.")

    iter_diag_dim = int(bitstring_matrix.shape[0])

    del expanded_list
    gc.collect()
    _ram("after rollback + gc")

    accepted_probs  = np.abs(eigenvector) ** 2
    accepted_bm_int = bitstring_matrix.astype(int)
    iter_n_above    = int(np.sum(
        (accepted_probs > PROB_THRESH) & ~np.all(accepted_bm_int == hf_row, axis=1)
    ))
    del accepted_probs, accepted_bm_int

    history["iteration"].append(ml_iter + 1)
    history["energy"].append(best_energy)
    history["diag_dimension"].append(iter_diag_dim)
    history["n_above_threshold"].append(iter_n_above)
    history["new_ml_dets"].append(len(new_determinants))
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)

    if stagnation >= CONV_WINDOW:
        print(f"\nConverged: energy change < {E_CONV:.0e} Ha for {CONV_WINDOW} "
              f"consecutive iterations. Stopping at iteration {ml_iter + 1}.")
        break

print(f"\nBest energy across all ML iterations: {best_energy}")
print(f"ML history saved to {history_path}")


import math
import multiprocessing as mp
import diagonalizer_dice_cpp as _ddc
import dice_core
from diagonalizer_dice_cpp import _interleave_parity_signs

# Semistochastic Epstein-Nesbet PT2, per Sharma, Holmes, Jeanmairet, Alavi,
# Umrigar, J. Chem. Theory Comput. 2017, 13, 1595-1604, Eqs. 8-11: a full
# deterministic sum over V at a loose threshold, plus a stochastic estimate
# of the tight-vs-loose difference from the same sampled batch.

if hasattr(np, "bitwise_count"):
    def _pc(x):
        """Vectorized popcount on a uint64 array."""
        x = np.ascontiguousarray(x, dtype=np.uint64)
        return np.bitwise_count(x).astype(np.int64)
else:
    def _pc(x):
        """Vectorized popcount on a uint64 array."""
        x = np.ascontiguousarray(x, dtype=np.uint64)
        return np.array([int(v).bit_count() for v in x.ravel()],
                        dtype=np.int64).reshape(x.shape)


def so_from_ab(alpha, beta, norb):
    """Pack alpha/beta spatial-occupancy ints into Dice spin-orbital ints."""
    alpha = np.ascontiguousarray(alpha, dtype=np.uint64)
    beta = np.ascontiguousarray(beta, dtype=np.uint64)
    so = np.zeros_like(alpha)
    one = np.uint64(1)
    for i in range(norb):
        bi = np.uint64(i)
        so |= ((alpha >> bi) & one) << np.uint64(2 * i)
        so |= ((beta >> bi) & one) << np.uint64(2 * i + 1)
    return so


def ab_from_so(so, norb):
    """Unpack Dice spin-orbital ints into (alpha, beta) spatial-occupancy ints."""
    so = np.ascontiguousarray(so, dtype=np.uint64)
    one = np.uint64(1)
    a = np.zeros_like(so)
    b = np.zeros_like(so)
    for i in range(norb):
        a |= ((so >> np.uint64(2 * i)) & one) << np.uint64(i)
        b |= ((so >> np.uint64(2 * i + 1)) & one) << np.uint64(i)
    return a, b


_TRIU_CACHE = {}


def _triu_pairs(n):
    """Memoized np.triu_indices(n, k=1)."""
    pair = _TRIU_CACHE.get(n)
    if pair is None:
        pair = np.triu_indices(n, k=1)
        _TRIU_CACHE[n] = pair
    return pair


def _excite(L, holes, parts):
    """Apply an ordered hole/particle excitation to determinant L, returning (sign, ext_so)."""
    one = np.uint64(1)
    shape = holes[0].shape
    d = np.full(shape, np.uint64(L), dtype=np.uint64)
    s = np.zeros(shape, dtype=np.int64)
    for h in holes:
        h64 = h.astype(np.uint64)
        mask = (one << h64) - one
        s ^= _pc(d & mask) & 1
        d &= ~(one << h64)
    for p in parts:
        p64 = p.astype(np.uint64)
        mask = (one << p64) - one
        s ^= _pc(d & mask) & 1
        d |= (one << p64)
    sign = 1.0 - 2.0 * s.astype(np.float64)
    return sign, d


def _connected_contributions(L, cI, hcore, eri, norb):
    """Generate all single/double excitations of L with c_I * <D_a|H|D_L>."""
    nso = 2 * norb
    occ = [b for b in range(nso) if (L >> b) & 1]
    virt = [b for b in range(nso) if not (L >> b) & 1]
    occ_a = [b for b in occ if b % 2 == 0]
    occ_b = [b for b in occ if b % 2 == 1]
    virt_a = [b for b in virt if b % 2 == 0]
    virt_b = [b for b in virt if b % 2 == 1]

    occ_sp = np.array([b >> 1 for b in occ], dtype=np.int64)
    coulomb_full = eri[:, :, occ_sp, occ_sp].sum(-1)
    occa_sp = np.array([b >> 1 for b in occ_a], dtype=np.int64)
    occb_sp = np.array([b >> 1 for b in occ_b], dtype=np.int64)
    exch_a = eri[:, occa_sp, occa_sp, :].sum(1) if occ_a else np.zeros_like(hcore)
    exch_b = eri[:, occb_sp, occb_sp, :].sum(1) if occ_b else np.zeros_like(hcore)

    es = []
    cs = []

    for occ_s, virt_s, exch_s in ((occ_a, virt_a, exch_a), (occ_b, virt_b, exch_b)):
        if not occ_s or not virt_s:
            continue
        no, nv = len(occ_s), len(virt_s)
        pso = np.repeat(np.array(occ_s, dtype=np.int64), nv)
        rso = np.tile(np.array(virt_s, dtype=np.int64), no)
        P = pso >> 1
        R = rso >> 1
        coulomb = coulomb_full[P, R] - eri[P, R, P, P]
        exch = exch_s[P, R] - eri[P, P, P, R]
        val = hcore[P, R] + coulomb - exch
        sign, ext = _excite(L, [pso], [rso])
        es.append(ext)
        cs.append(cI * sign * val)

    def _add_doubles(h1, h2, p1, p2):
        H1, H2, P1, P2 = h1 >> 1, h2 >> 1, p1 >> 1, p2 >> 1
        sh1, sh2, sp1, sp2 = h1 & 1, h2 & 1, p1 & 1, p2 & 1
        t1 = ((sh1 == sp1) & (sh2 == sp2)).astype(np.float64) * eri[H1, P1, H2, P2]
        t2 = ((sh1 == sp2) & (sh2 == sp1)).astype(np.float64) * eri[H1, P2, H2, P1]
        val = t1 - t2
        sign, ext = _excite(L, [h1, h2], [p1, p2])
        es.append(ext)
        cs.append(cI * (-sign) * val)

    for occ_s, virt_s in ((occ_a, virt_a), (occ_b, virt_b)):
        if len(occ_s) >= 2 and len(virt_s) >= 2:
            oi, oj = _triu_pairs(len(occ_s))
            vi, vj = _triu_pairs(len(virt_s))
            os_ = np.array(occ_s, dtype=np.int64)
            vs_ = np.array(virt_s, dtype=np.int64)
            h1 = np.repeat(os_[oi], len(vi))
            h2 = np.repeat(os_[oj], len(vi))
            p1 = np.tile(vs_[vi], len(oi))
            p2 = np.tile(vs_[vj], len(oi))
            _add_doubles(h1, h2, p1, p2)

    if occ_a and occ_b and virt_a and virt_b:
        oa = np.array(occ_a, dtype=np.int64)
        ob = np.array(occ_b, dtype=np.int64)
        va = np.array(virt_a, dtype=np.int64)
        vb = np.array(virt_b, dtype=np.int64)
        HA = np.repeat(oa, len(ob))
        HB = np.tile(ob, len(oa))
        nh = len(HA)
        PA = np.repeat(va, len(vb))
        PB = np.tile(vb, len(va))
        npp = len(PA)
        HA = np.repeat(HA, npp); HB = np.repeat(HB, npp)
        PA = np.tile(PA, nh);   PB = np.tile(PB, nh)
        h1 = np.minimum(HA, HB); h2 = np.maximum(HA, HB)
        p1 = np.minimum(PA, PB); p2 = np.maximum(PA, PB)
        _add_doubles(h1, h2, p1, p2)

    if not es:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64))
    return np.concatenate(es), np.concatenate(cs)


def _hij_scalar(so_i, so_j, hcore, eri, norb):
    """Scalar <D_i|H|D_j> for i != j, used only by the self-check."""
    xor = so_i ^ so_j
    rank = bin(xor).count("1") // 2
    if rank == 0 or rank > 2:
        return 0.0
    holes = sorted(b for b in range(2 * norb) if (so_i >> b) & 1 and (xor >> b) & 1)
    parts = sorted(b for b in range(2 * norb) if (so_j >> b) & 1 and (xor >> b) & 1)

    def phase(holes, parts):
        d = so_i
        s = 0
        for h in holes:
            s ^= bin(d & ((1 << h) - 1)).count("1") & 1
            d &= ~(1 << h)
        for p in parts:
            s ^= bin(d & ((1 << p) - 1)).count("1") & 1
            d |= (1 << p)
        return 1 - 2 * s

    if rank == 1:
        p, r = holes[0], parts[0]
        if (p & 1) != (r & 1):
            return 0.0
        P, R = p >> 1, r >> 1
        occ = [b for b in range(2 * norb) if (so_i >> b) & 1]
        val = hcore[P, R]
        for q in occ:
            if q == p:
                continue
            Q = q >> 1
            val += eri[P, R, Q, Q]
            if (q & 1) == (p & 1):
                val -= eri[P, Q, Q, R]
        return phase(holes, parts) * val

    h1, h2 = holes
    p1, p2 = parts
    H1, H2, P1, P2 = h1 >> 1, h2 >> 1, p1 >> 1, p2 >> 1
    val = 0.0
    if (h1 & 1) == (p1 & 1) and (h2 & 1) == (p2 & 1):
        val += eri[H1, P1, H2, P2]
    if (h1 & 1) == (p2 & 1) and (h2 & 1) == (p1 & 1):
        val -= eri[H1, P2, H2, P1]
    return -phase(holes, parts) * val


def verify_matrix_elements(hcore, eri, norb, n_alpha, n_beta, dice_core,
                           n_test=60, seed=0):
    """Validate the Slater-Condon implementation against dice_core; raise on mismatch."""
    rng = np.random.default_rng(seed)
    seen = set()
    while len(seen) < n_test:
        a = tuple(sorted(rng.choice(norb, n_alpha, replace=False).tolist()))
        b = tuple(sorted(rng.choice(norb, n_beta, replace=False).tolist()))
        seen.add((a, b))
    dets = list(seen)
    alpha = np.array([sum(1 << i for i in a) for a, _ in dets], dtype=np.uint64)
    beta = np.array([sum(1 << i for i in b) for _, b in dets], dtype=np.uint64)

    Hd = np.asarray(dice_core.build_dense_hamiltonian(alpha, beta, hcore, eri))
    so = so_from_ab(alpha, beta, norb)

    maxerr = 0.0
    for i in range(len(so)):
        for j in range(len(so)):
            if i == j:
                continue
            mine = _hij_scalar(int(so[i]), int(so[j]), hcore, eri, norb)
            maxerr = max(maxerr, abs(mine - Hd[i, j]))

    diag = np.asarray(dice_core.diagonal_energies(alpha, beta, hcore, eri))
    diagerr = float(np.max(np.abs(np.diag(Hd) - diag)))

    if maxerr > 1e-8 or diagerr > 1e-8:
        raise RuntimeError(
            f"PT2 matrix-element self-check FAILED "
            f"(off-diagonal |dHij|={maxerr:.2e}, diagonal |dHii|={diagerr:.2e}). "
            "The Slater-Condon phase/value convention does not match dice_core; "
            "PT2 aborted to avoid a silently wrong correction."
        )
    print(f"  self-check passed: max off-diag |dHij|={maxerr:.2e}, "
          f"diag |dHii|={diagerr:.2e}  ({len(so)} dets, {len(so)*(len(so)-1)} pairs)")


def _det_chunk_worker(args):
    chunk_so, chunk_ci, hcore, eri, norb, eps2 = args
    ext_acc, con_acc = [], []
    for so_i, cI in zip(chunk_so, chunk_ci):
        if cI == 0.0:
            continue
        ext, con = _connected_contributions(int(so_i), cI, hcore, eri, norb)
        if ext.size:
            keep = np.abs(con) > eps2
            if keep.any():
                ext_acc.append(ext[keep])
                con_acc.append(con[keep])
    if not ext_acc:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64)
    ext_all = np.concatenate(ext_acc)
    con_all = np.concatenate(con_acc)
    uniq, inv = np.unique(ext_all, return_inverse=True)
    num = np.bincount(inv.ravel(), weights=con_all, minlength=len(uniq))
    return uniq, num


def _compact(ext_list, num_list):
    ext_all = np.concatenate(ext_list)
    num_all = np.concatenate(num_list)
    uniq, inv = np.unique(ext_all, return_inverse=True)
    num = np.bincount(inv.ravel(), weights=num_all, minlength=len(uniq))
    return [uniq], [num]


def deterministic_pt2_full(ref_so, ci, hcore, eri, e_var, norb, diag_fn, eps2,
                            n_workers=None, chunk_size=20000,
                            compact_every=10, verbose=True):
    """Full deterministic sum over every determinant in V, screened at eps2 (Eq. 9)."""
    ref_so = np.ascontiguousarray(ref_so, dtype=np.uint64)
    ci = np.ascontiguousarray(ci, dtype=np.float64)
    ref_sorted = np.sort(ref_so)
    n = len(ci)

    chunks = [(ref_so[s:s + chunk_size], ci[s:s + chunk_size], hcore, eri, norb, eps2)
              for s in range(0, n, chunk_size)]

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    ext_acc, num_acc = [], []
    if n_workers > 1 and len(chunks) > 1:
        # Fork explicitly: workers only touch plain numpy arrays, never CUDA,
        # and fork (unlike spawn/forkserver) doesn't re-import this
        # top-level, __main__-guard-less script in the child.
        with mp.get_context("fork").Pool(n_workers) as pool:
            for i, (uniq, num) in enumerate(pool.imap_unordered(_det_chunk_worker, chunks)):
                if uniq.size:
                    ext_acc.append(uniq)
                    num_acc.append(num)
                if compact_every and (i + 1) % compact_every == 0 and ext_acc:
                    ext_acc, num_acc = _compact(ext_acc, num_acc)
                if verbose and (i + 1) % max(1, len(chunks) // 20) == 0:
                    print(f"    [det-full] {i + 1}/{len(chunks)} chunks")
    else:
        for i, c in enumerate(chunks):
            uniq, num = _det_chunk_worker(c)
            if uniq.size:
                ext_acc.append(uniq)
                num_acc.append(num)
            if compact_every and (i + 1) % compact_every == 0 and ext_acc:
                ext_acc, num_acc = _compact(ext_acc, num_acc)

    empty_info = {"n_external": 0}
    if not ext_acc:
        return 0.0, empty_info

    ext_acc, num_acc = _compact(ext_acc, num_acc)
    uniq, Nd = ext_acc[0], num_acc[0]

    pos = np.clip(np.searchsorted(ref_sorted, uniq), 0, len(ref_sorted) - 1)
    keep = ref_sorted[pos] != uniq
    uniq, Nd = uniq[keep], Nd[keep]

    if uniq.size == 0:
        return 0.0, empty_info

    alpha, beta = ab_from_so(uniq, norb)
    haa = np.asarray(diag_fn(np.ascontiguousarray(alpha), np.ascontiguousarray(beta), hcore, eri))
    denom = e_var - haa
    e_det = float(np.sum(Nd ** 2 / denom))
    return e_det, {"n_external": int(uniq.size)}


def _batch_generate(sel, w, p_sel, ref_so, ci, hcore, eri, norb):
    """Unscreened connections for the sampled generators in this batch."""
    data, w_out, p_out = [], [], []
    for idx, wi, pi in zip(sel, w, p_sel):
        cI = ci[idx]
        if cI == 0.0:
            continue
        ext, con = _connected_contributions(int(ref_so[idx]), cI, hcore, eri, norb)
        if ext.size:
            data.append((ext, con))
            w_out.append(wi)
            p_out.append(pi)
    return data, np.array(w_out), np.array(p_out)


def _estimator_from_batch(batch_data, w, p_sel, eps2, Nd):
    """Eq. 10: S1_a = sum_i w_i c_i H_ai/p_i, S2_a = sum_i (w_i(Nd-1)/p_i - w_i^2/p_i^2) c_i^2 H_ai^2."""
    ext_chunks, s1_chunks, s2_chunks = [], [], []
    for (ext, con), wi, pi in zip(batch_data, w, p_sel):
        keep = np.abs(con) > eps2
        if not keep.any():
            continue
        ext_k, con_k = ext[keep], con[keep]
        ext_chunks.append(ext_k)
        s1_chunks.append(con_k * (wi / pi))
        coef2 = wi * (Nd - 1) / pi - (wi * wi) / (pi * pi)
        s2_chunks.append((con_k * con_k) * coef2)
    if not ext_chunks:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64))
    ext_all = np.concatenate(ext_chunks)
    s1_all = np.concatenate(s1_chunks)
    s2_all = np.concatenate(s2_chunks)
    uniq, inv = np.unique(ext_all, return_inverse=True)
    inv = inv.ravel()
    S1 = np.bincount(inv, weights=s1_all, minlength=len(uniq))
    S2 = np.bincount(inv, weights=s2_all, minlength=len(uniq))
    return uniq, S1, S2


def _batch_value(uniq, S1, S2, ref_sorted, haa_lookup, e_var, Nd):
    if uniq.size == 0:
        return 0.0
    pos = np.clip(np.searchsorted(ref_sorted, uniq), 0, len(ref_sorted) - 1)
    keepV = ref_sorted[pos] != uniq
    uniq, S1, S2 = uniq[keepV], S1[keepV], S2[keepV]
    if uniq.size == 0:
        return 0.0
    haa = haa_lookup(uniq)
    denom = e_var - haa
    val = (S1 ** 2 + S2) / (Nd * (Nd - 1))
    return float(np.sum(val / denom))


def semistochastic_pt2_holmes(ref_so, ci, hcore, eri, e_var, norb, diag_fn,
                               eps2=1e-8, eps2_d=5e-6, Nd=200, n_batches=20,
                               seed=0, n_workers=None,
                               chunk_size=20000, verbose=True):
    """Semistochastic PT2 per Sharma et al. 2017, Eq. 11: full deterministic
    sweep at the loose threshold eps2_d, plus a stochastic tight-vs-loose
    correction evaluated on the same sampled batch."""
    if Nd < 2:
        raise ValueError("Nd must be >= 2 (Eq. 10 needs N_d(N_d-1) in the denominator).")
    if n_batches < 2:
        raise ValueError("n_batches must be >= 2 to estimate a standard error.")

    ref_so = np.ascontiguousarray(ref_so, dtype=np.uint64)
    ci = np.ascontiguousarray(ci, dtype=np.float64)
    ref_sorted = np.sort(ref_so)
    Nv = len(ci)

    if verbose:
        print(f"  [deterministic] full sweep over {Nv:,} reference determinants "
              f"at eps2_d={eps2_d:g} ...")
    e_det_loose, det_info = deterministic_pt2_full(
        ref_so, ci, hcore, eri, e_var, norb, diag_fn, eps2_d,
        n_workers=n_workers, chunk_size=chunk_size, verbose=verbose)
    if verbose:
        print(f"  [deterministic] E_PT2^D[eps2_d] = {e_det_loose:+.8f}  "
              f"over {det_info['n_external']:,} externals")

    abs_c = np.abs(ci)
    p_all = abs_c / abs_c.sum()
    cdf_all = np.cumsum(p_all)
    cdf_all[-1] = 1.0
    rng = np.random.default_rng(seed)

    haa_cache = {}

    def haa_lookup(uniq):
        missing_mask = np.array([int(u) not in haa_cache for u in uniq])
        if missing_mask.any():
            missing = uniq[missing_mask]
            a, b = ab_from_so(missing, norb)
            vals = np.asarray(diag_fn(np.ascontiguousarray(a), np.ascontiguousarray(b), hcore, eri))
            for u, v in zip(missing, vals):
                haa_cache[int(u)] = float(v)
        return np.array([haa_cache[int(u)] for u in uniq])

    diffs = np.empty(n_batches, dtype=np.float64)
    for b in range(n_batches):
        u = rng.random(Nd)
        jl = np.clip(np.searchsorted(cdf_all, u, side="right"), 0, Nv - 1)
        sel, w = np.unique(jl, return_counts=True)
        w = w.astype(np.float64)
        p_sel = p_all[sel]

        batch_data, w_aligned, p_aligned = _batch_generate(
            sel, w, p_sel, ref_so, ci, hcore, eri, norb)

        uq_t, S1_t, S2_t = _estimator_from_batch(batch_data, w_aligned, p_aligned, eps2, Nd)
        uq_l, S1_l, S2_l = _estimator_from_batch(batch_data, w_aligned, p_aligned, eps2_d, Nd)

        val_tight = _batch_value(uq_t, S1_t, S2_t, ref_sorted, haa_lookup, e_var, Nd)
        val_loose = _batch_value(uq_l, S1_l, S2_l, ref_sorted, haa_lookup, e_var, Nd)
        diffs[b] = val_tight - val_loose

        if verbose:
            rm = diffs[:b + 1].mean()
            rse = (diffs[:b + 1].std(ddof=1) / math.sqrt(b + 1)) if b >= 1 else float("nan")
            print(f"  batch {b + 1:>3}/{n_batches}  correction={diffs[b]:+.6f}  "
                  f"E_PT2={e_det_loose + rm:+.6f} +/- {rse:.6f}")

    stoch_mean = float(diffs.mean())
    stoch_se = float(diffs.std(ddof=1) / math.sqrt(n_batches))
    e_pt2 = e_det_loose + stoch_mean

    info = {
        "eps2_tight": eps2,
        "eps2_loose": eps2_d,
        "Nd": Nd,
        "n_batches": n_batches,
        "e_pt2_deterministic_loose": e_det_loose,
        "e_pt2_stochastic_correction_mean": stoch_mean,
        "n_external_deterministic": det_info["n_external"],
        "seed": seed,
        "batch_values": diffs.tolist(),
    }
    return e_pt2, stoch_se, info


PT2_EPS2       = 1e-8
PT2_EPS2_LOOSE = 5e-6
PT2_ND         = 200
PT2_N_BATCHES  = 20
PT2_SEED       = 0

_norb  = _ddc._NORB
_hcore = _ddc._HCORE
_eri   = _ddc._ERI_CHEM
_na, _nb = num_particles

_e_var_elec = best_energy - nuclear_repulsion_energy - fc_shift

_bm      = best_bitstring_matrix.astype(np.uint64)
_nq      = _bm.shape[1]
_shifts  = np.arange(_nq - 1, -1, -1, dtype=np.uint64)
_ref_int = (_bm << _shifts).sum(axis=1).astype(np.uint64)
_alpha   = _ref_int & np.uint64((1 << _norb) - 1)
_beta    = _ref_int >> np.uint64(_norb)
_ci_jw   = np.asarray(best_eigenvector, dtype=np.float64)
_nz      = np.abs(_ci_jw) > 0.0
_alpha, _beta, _ci_jw = _alpha[_nz], _beta[_nz], _ci_jw[_nz]
_ci_dice = _ci_jw * _interleave_parity_signs(_alpha, _beta, _norb)
_ref_so  = so_from_ab(_alpha, _beta, _norb)

_ckpt_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(_ckpt_dir, exist_ok=True)
_ckpt_path = os.path.join(_ckpt_dir, f"{mol_name}_hardware.npz")
np.savez(
    _ckpt_path,
    ref_so=_ref_so, ci_dice=_ci_dice,
    hcore=_hcore, eri=_eri, norb=np.int64(_norb),
    n_alpha=np.int64(_na), n_beta=np.int64(_nb),
    best_energy=np.float64(best_energy),
    e_var_elec=np.float64(_e_var_elec),
    nuclear_repulsion_energy=np.float64(nuclear_repulsion_energy),
    fc_shift=np.float64(fc_shift),
)
print(f"\nPT2 checkpoint saved to {_ckpt_path} ({len(_ref_so):,} determinants)")

print("\n=== Semistochastic Epstein-Nesbet PT2 (single, full wavefunction) ===")
print(f"  reference determinants: {len(_ref_so):,}  "
      f"(eps2={PT2_EPS2:g}, eps2_d={PT2_EPS2_LOOSE:g}, "
      f"{PT2_N_BATCHES}x{PT2_ND} sampled per batch)")
print("  verifying Slater-Condon matrix elements against dice_core ...")
verify_matrix_elements(_hcore, _eri, _norb, int(_na), int(_nb), dice_core)

_e_pt2, _e_pt2_stderr, _pt2_info = semistochastic_pt2_holmes(
    _ref_so, _ci_dice, _hcore, _eri, _e_var_elec, _norb,
    dice_core.diagonal_energies,
    eps2=PT2_EPS2, eps2_d=PT2_EPS2_LOOSE, Nd=PT2_ND,
    n_batches=PT2_N_BATCHES, seed=PT2_SEED, verbose=True,
)

_e_var_total  = best_energy
_e_corr_total = _e_var_total + _e_pt2
print("=" * 76)
print(f"  E_var          (total) = {_e_var_total:.8f} Ha")
print(f"  E_PT2                   = {_e_pt2:+.8f} +/- {_e_pt2_stderr:.8f} Ha")
print(f"  E_var + E_PT2  (total) = {_e_corr_total:.8f} Ha")
print("=" * 76)

history["pt2"] = {
    "e_var_total": _e_var_total,
    "e_pt2": float(_e_pt2),
    "e_pt2_stderr": float(_e_pt2_stderr),
    "e_var_plus_pt2_total": _e_corr_total,
    "eps2": PT2_EPS2,
    "eps2_loose": PT2_EPS2_LOOSE,
    "Nd": PT2_ND,
    "n_batches": PT2_N_BATCHES,
    "seed": PT2_SEED,
    "e_pt2_deterministic_loose": float(_pt2_info.get("e_pt2_deterministic_loose", 0.0)),
    "n_external_deterministic": int(_pt2_info.get("n_external_deterministic", 0)),
}
with open(history_path, "w") as fh:
    json.dump(history, fh, indent=2)
print(f"PT2 result appended to history file {history_path}")
