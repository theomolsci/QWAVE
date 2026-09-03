/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  Hmult2: y = H.x from the half-stored (lower-triangle) sparse Hamiltonian.
  This is Dice's serial "< 10 million dets" path (accumulate into a temporary
  then copy out), parallelized over interleaved row blocks with std::thread.
  Each thread accumulates into its own full-length buffer (the same pattern
  Dice uses across MPI ranks with MPI_Reduce); buffers are then reduced in
  fixed thread order, so the result is deterministic for a given nThreads.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
#include <algorithm>
#include <thread>
#include <vector>
#include "Determinants.h"
#include "SHCImakeHamiltonian.h"
#include "global.h"

using namespace std;
using namespace SHCImakeHamiltonian;

struct Hmult2 {
  SparseHam& sparseHam;
  int nThreads;
  std::vector<std::vector<CItype>> ytemp;  // one accumulator per thread

  Hmult2(SparseHam& p_sparseHam, int p_nThreads)
      : sparseHam(p_sparseHam), nThreads(p_nThreads < 1 ? 1 : p_nThreads) {}

  //===========================================================================
  void operator()(CItype* x, CItype* y) {
    /*!
    Calculate y = H.x from the sparse Hamiltonian (y is overwritten).
    */
    size_t numDets = sparseHam.connections.size();
    int T = nThreads;

    if ((int)ytemp.size() != T || (T > 0 && ytemp[0].size() != numDets))
      ytemp.assign(T, std::vector<CItype>(numDets, 0.0));
    else
      for (auto& v : ytemp) std::fill(v.begin(), v.end(), 0.0);

    auto work = [&](int t) {
      std::vector<CItype>& yt = ytemp[t];
      for (size_t i = t; i < numDets; i += T) {
        const std::vector<int>& conn = sparseHam.connections[i];
        const std::vector<CItype>& hel = sparseHam.Helements[i];
        CItype yi = 0.0;
        for (size_t j = 0; j < conn.size(); j++) {
          CItype hij = hel[j];
          int J = conn[j];
          yi += hij * x[J];
          if (J != (int)i) yt[J] += hij * x[i];
        }
        yt[i] += yi;
      }
    };

    {
      std::vector<std::thread> threads;
      for (int t = 1; t < T; t++) threads.emplace_back(work, t);
      work(0);
      for (auto& th : threads) th.join();
    }

    // reduce per-thread buffers into y (fixed order -> deterministic)
    auto reduceWork = [&](int t) {
      size_t chunk = (numDets + T - 1) / T;
      size_t lo = (size_t)t * chunk, hi = std::min(numDets, lo + chunk);
      for (size_t k = lo; k < hi; k++) {
        CItype s = 0.0;
        for (int tt = 0; tt < T; tt++) s += ytemp[tt][k];
        y[k] = s;
      }
    };

    {
      std::vector<std::thread> threads;
      for (int t = 1; t < T; t++) threads.emplace_back(reduceWork, t);
      reduceWork(0);
      for (auto& th : threads) th.join();
    }
  }  // operator
};

#endif
