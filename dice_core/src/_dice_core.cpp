/*
  _dice_core: pybind11 bindings around the machinery extracted from Dice
  (https://github.com/caleb-johnson/Dice), GPLv3.

  Diagonalizes a molecular Hamiltonian in a FIXED list of Slater determinants
  (no heat-bath/HCI expansion, no Cartesian product): the variational space is
  exactly the n (alpha, beta) pairs passed in, in that order.
*/
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "SHCImakeHamiltonian.h"
#include "global.h"
#include "integral.h"

namespace py = pybind11;
using namespace SHCImakeHamiltonian;

namespace {

int resolveThreads(int n_threads) {
  if (n_threads > 0) return n_threads;
  unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1 : (int)hc;
}

// Initialize the Dice global statics for a given number of SPATIAL orbitals.
void initStatics(int norb) {
  Determinant::norbs = 2 * norb;  // spin orbitals
  HalfDet::norbs = 2 * norb;      // spin orbitals
  Determinant::EffDetLen = (2 * norb + 63) / 64;
  Determinant::Trev = 0;
  if (Determinant::EffDetLen > DetLen)
    throw std::invalid_argument("norb too large for compiled DetLen");
}

// Parse + validate the determinant arrays; returns Dice-packed Determinants.
// Dice bit convention: spin orbital 2*i = alpha_i, 2*i+1 = beta_i.
std::vector<Determinant> buildDets(py::array_t<uint64_t> alpha_dets,
                                   py::array_t<uint64_t> beta_dets, int norb) {
  auto a = alpha_dets.unchecked<1>();
  auto b = beta_dets.unchecked<1>();
  if (a.shape(0) != b.shape(0))
    throw std::invalid_argument(
        "alpha_dets and beta_dets must have the same length");
  ssize_t n = a.shape(0);
  if (n < 1) throw std::invalid_argument("need at least one determinant");
  if (norb < 1 || norb > 64)
    throw std::invalid_argument("norb must be between 1 and 64");

  uint64_t allowed = (norb == 64) ? ~0ULL : ((1ULL << norb) - 1);
  int nalpha = __builtin_popcountll(a(0));
  int nbeta = __builtin_popcountll(b(0));

  std::vector<Determinant> dets((size_t)n);
  std::vector<std::pair<uint64_t, uint64_t>> pairs((size_t)n);
  for (ssize_t k = 0; k < n; k++) {
    uint64_t av = a(k), bv = b(k);
    if ((av & ~allowed) || (bv & ~allowed))
      throw std::invalid_argument(
          "determinant " + std::to_string(k) +
          " has occupied orbitals beyond norb=" + std::to_string(norb));
    if (__builtin_popcountll(av) != nalpha ||
        __builtin_popcountll(bv) != nbeta)
      throw std::invalid_argument(
          "determinant " + std::to_string(k) +
          " does not have the same (n_alpha, n_beta) as determinant 0");
    pairs[k] = std::make_pair(av, bv);
    Determinant& d = dets[k];
    for (int i = 0; i < norb; i++) {
      if ((av >> i) & 1ULL) d.setocc(2 * i, true);
      if ((bv >> i) & 1ULL) d.setocc(2 * i + 1, true);
    }
  }

  // duplicates would silently corrupt the half-string helper machinery
  std::sort(pairs.begin(), pairs.end());
  if (std::adjacent_find(pairs.begin(), pairs.end()) != pairs.end())
    throw std::invalid_argument("duplicate (alpha, beta) determinant pairs");

  return dets;
}

// Fill oneInt/twoInt exactly the way Dice's FCIDUMP reader does, but from
// numpy arrays. hcore: (norb, norb); eri: (norb,)*4 CHEMIST notation (ij|kl).
void fillIntegrals(py::array_t<double> hcore, py::array_t<double> eri,
                   int norb, oneInt& I1, twoInt& I2) {
  if (hcore.ndim() != 2 || hcore.shape(0) != norb || hcore.shape(1) != norb)
    throw std::invalid_argument("hcore must have shape (norb, norb)");
  if (eri.ndim() != 4 || eri.shape(0) != norb || eri.shape(1) != norb ||
      eri.shape(2) != norb || eri.shape(3) != norb)
    throw std::invalid_argument(
        "eri must have shape (norb, norb, norb, norb)");

  auto h = hcore.unchecked<2>();
  auto v = eri.unchecked<4>();

  I1.norbs = 2 * norb;
  I1.store.assign((size_t)(2 * norb) * (2 * norb), 0.0);
  for (int a = 0; a < norb; a++)
    for (int b = 0; b < norb; b++) {
      I1(2 * a, 2 * b) = h(a, b);          // alpha,alpha
      I1(2 * a + 1, 2 * b + 1) = h(a, b);  // beta,beta
    }

  I2.ksym = false;
  I2.norbs = norb;
  size_t npair = (size_t)norb * (norb + 1) / 2;
  I2.store.assign(npair * (npair + 1) / 2, 0.0);
  for (int i = 0; i < norb; i++)
    for (int j = 0; j < norb; j++)
      for (int k = 0; k < norb; k++)
        for (int l = 0; l < norb; l++)
          I2(2 * i, 2 * j, 2 * k, 2 * l) = v(i, j, k, l);

  I2.Direct = MatrixXd::Zero(norb, norb);
  I2.Exchange = MatrixXd::Zero(norb, norb);
  for (int i = 0; i < norb; i++)
    for (int j = 0; j < norb; j++) {
      I2.Direct(i, j) = I2(2 * i, 2 * i, 2 * j, 2 * j);
      I2.Exchange(i, j) = I2(2 * i, 2 * j, 2 * j, 2 * i);
    }
}

}  // namespace

//=============================================================================
py::tuple solve_fixed_basis(py::array_t<uint64_t> alpha_dets,
                            py::array_t<uint64_t> beta_dets,
                            py::array_t<double> hcore, py::array_t<double> eri,
                            double davidson_tol, int max_iter,
                            int max_subspace, int n_threads, bool verbose) {
  int norb = (int)hcore.shape(0);
  initStatics(norb);
  std::vector<Determinant> dets = buildDets(alpha_dets, beta_dets, norb);
  int n = (int)dets.size();

  oneInt I1;
  twoInt I2;
  fillIntegrals(hcore, eri, norb, I1, I2);
  double coreE = 0.0;

  int nT = resolveThreads(n_threads);
  int maxCopies = std::min((long)n, (long)std::max(max_subspace, 5));
  if (max_iter < 1) throw std::invalid_argument("max_iter must be >= 1");

  double energy = 0.0;
  int numIter = 0;
  std::vector<double> ci((size_t)n, 0.0);

  {
    py::gil_scoped_release release;

    HamHelpers2 helpers;
    helpers.PopulateHelpers(dets.data(), n, 0);
    SparseHam ham;
    ham.makeFromHelper(helpers, dets.data(), n, I1, I2, coreE, nT);

    MatrixXx diag(n, 1);
    for (int k = 0; k < n; k++) diag(k, 0) = ham.Helements[k][0];

    std::vector<MatrixXx> x0(1, MatrixXx::Zero(n, 1));
    int kmin = 0;
    for (int k = 1; k < n; k++)
      if (diag(k, 0) < diag(kmin, 0)) kmin = k;
    x0[0](kmin, 0) = 1.0;

    Hmult2 H(ham, nT);
    std::vector<double> eroots = davidson(H, x0, diag, maxCopies, davidson_tol,
                                          numIter, max_iter, verbose);
    energy = eroots[0];
    for (int k = 0; k < n; k++) ci[k] = x0[0](k, 0);
  }

  py::array_t<double> ciArr((ssize_t)n);
  std::copy(ci.begin(), ci.end(), ciArr.mutable_data());
  return py::make_tuple(energy, ciArr, numIter);
}

//=============================================================================
py::array_t<double> diagonal_energies(py::array_t<uint64_t> alpha_dets,
                                      py::array_t<uint64_t> beta_dets,
                                      py::array_t<double> hcore,
                                      py::array_t<double> eri, int n_threads) {
  int norb = (int)hcore.shape(0);
  initStatics(norb);
  std::vector<Determinant> dets = buildDets(alpha_dets, beta_dets, norb);
  int n = (int)dets.size();

  oneInt I1;
  twoInt I2;
  fillIntegrals(hcore, eri, norb, I1, I2);
  double coreE = 0.0;
  int nT = resolveThreads(n_threads);

  std::vector<double> energies((size_t)n, 0.0);
  {
    py::gil_scoped_release release;
    std::atomic<int> next(0);
    auto work = [&]() {
      int k;
      while ((k = next.fetch_add(256)) < n) {
        int kend = std::min(k + 256, n);
        for (; k < kend; k++) energies[k] = dets[k].Energy(I1, I2, coreE);
      }
    };
    std::vector<std::thread> threads;
    for (int t = 1; t < nT; t++) threads.emplace_back(work);
    work();
    for (auto& th : threads) th.join();
  }

  py::array_t<double> out((ssize_t)n);
  std::copy(energies.begin(), energies.end(), out.mutable_data());
  return out;
}

//=============================================================================
py::array_t<double> build_dense_hamiltonian(py::array_t<uint64_t> alpha_dets,
                                            py::array_t<uint64_t> beta_dets,
                                            py::array_t<double> hcore,
                                            py::array_t<double> eri) {
  int norb = (int)hcore.shape(0);
  initStatics(norb);
  std::vector<Determinant> dets = buildDets(alpha_dets, beta_dets, norb);
  int n = (int)dets.size();
  if (n > 4000)
    throw std::invalid_argument(
        "build_dense_hamiltonian refuses n > 4000 determinants");

  oneInt I1;
  twoInt I2;
  fillIntegrals(hcore, eri, norb, I1, I2);
  double coreE = 0.0;

  std::vector<double> Hd((size_t)n * n, 0.0);
  {
    py::gil_scoped_release release;
    for (int i = 0; i < n; i++) {
      Hd[(size_t)i * n + i] = dets[i].Energy(I1, I2, coreE);
      for (int j = 0; j < i; j++) {
        if (!dets[i].connected(dets[j])) continue;
        size_t orbDiff;
        double hij = Hij(dets[i], dets[j], I1, I2, coreE, orbDiff);
        Hd[(size_t)i * n + j] = hij;
        Hd[(size_t)j * n + i] = hij;
      }
    }
  }

  py::array_t<double> out({(ssize_t)n, (ssize_t)n});
  std::copy(Hd.begin(), Hd.end(), out.mutable_data());
  return out;
}

//=============================================================================
PYBIND11_MODULE(_dice_core, m) {
  m.doc() =
      "Fixed-basis determinant CI diagonalizer extracted from Dice "
      "(caleb-johnson/Dice, GPLv3). No HCI determinant-space expansion: the "
      "variational space is exactly the (alpha, beta) pairs passed in.";

  m.def("solve_fixed_basis", &solve_fixed_basis, py::arg("alpha_dets"),
        py::arg("beta_dets"), py::arg("hcore"), py::arg("eri"),
        py::arg("davidson_tol") = 1e-5, py::arg("max_iter") = 100,
        py::arg("max_subspace") = 40, py::arg("n_threads") = 0,
        py::arg("verbose") = true,
        "Diagonalize H in the fixed (alpha,beta) determinant basis.\n"
        "Returns (energy, ci, n_iter); energy is the pure active-space\n"
        "electronic energy (core energy = 0), ci is in input det order.");

  m.def("diagonal_energies", &diagonal_energies, py::arg("alpha_dets"),
        py::arg("beta_dets"), py::arg("hcore"), py::arg("eri"),
        py::arg("n_threads") = 0,
        "<D_k|H|D_k> for each determinant via Determinant::Energy.");

  m.def("build_dense_hamiltonian", &build_dense_hamiltonian,
        py::arg("alpha_dets"), py::arg("beta_dets"), py::arg("hcore"),
        py::arg("eri"),
        "Dense (n, n) Hamiltonian, debug/test helper; refuses n > 4000.");
}
