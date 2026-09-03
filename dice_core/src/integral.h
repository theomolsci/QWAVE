/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  Trimmed: only oneInt and twoInt kept; the FCIDUMP reader, heat-bath integral
  classes, boost serialization and the boost::interprocess shared-memory
  backing for twoInt::store are removed.  twoInt::store is now a plain
  std::vector<double> on the heap, but the *layout* is exactly Dice's:

   - oneInt: dense (2*norb_spatial)^2 array over SPIN orbitals, store[i*N+j];
     filled with I1(2a,2b) = I1(2a+1,2b+1) = hcore(a,b).
   - twoInt: 8-fold-symmetric packed array over SPATIAL orbitals in CHEMIST
     notation: I2(i,j,k,l) (spin-orbital indices) = (ij|kl) with
     IJ = max(I,J)(max(I,J)+1)/2 + min(I,J) etc., zero unless the spins of
     (i,j) and of (k,l) match.
   - Direct(I,J) = (II|JJ), Exchange(I,J) = (IJ|JI), spatial indices.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef INTEGRAL_HEADER_H
#define INTEGRAL_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "global.h"

using namespace std;
using namespace Eigen;

class oneInt {
 public:
  std::vector<CItype> store;
  int norbs;  // number of SPIN orbitals
  // I explicitly store all elements of the matrix
  // so for normal operator if i and j dont have the same spin
  // then it will just return zero.
  inline CItype& operator()(int i, int j) { return store.at(i * norbs + j); }
};

class twoInt {
 public:
  std::vector<double> store;
  MatrixXd Direct, Exchange;
  static thread_local double zero;
  size_t norbs;  // number of SPATIAL orbitals
  bool ksym;
  twoInt() : norbs(0), ksym(false) {}
  inline double& operator()(int i, int j, int k, int l) {
    zero = 0.0;
    if (!((i % 2 == j % 2) && (k % 2 == l % 2))) return zero;
    int I = i / 2;
    int J = j / 2;
    int K = k / 2;
    int L = l / 2;

    if (!ksym) {
      int IJ = max(I, J) * (max(I, J) + 1) / 2 + min(I, J);
      int KL = max(K, L) * (max(K, L) + 1) / 2 + min(K, L);
      int A = max(IJ, KL), B = min(IJ, KL);
      return store[A * (size_t)(A + 1) / 2 + B];
    } else {
      int IJ = I * norbs + J, KL = K * norbs + L;
      int A = max(IJ, KL), B = min(IJ, KL);
      return store[A * (size_t)(A + 1) / 2 + B];
    }
  }
};

#endif
