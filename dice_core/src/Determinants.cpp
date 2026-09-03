/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  Trimmed: only Energy / Hij_1Excite / Hij_2Excite / Hij retained (the
  Slater-Condon machinery actually used to build the Hamiltonian).  These
  bodies are Dice's, unmodified except for removal of the Complex branches.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#include "Determinants.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "integral.h"
using namespace std;

int HalfDet::norbs = 1;      // spin orbitals
int Determinant::norbs = 1;  // spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;  // Time reversal: always off in this extraction

//=============================================================================
double Determinant::Energy(oneInt& I1, twoInt& I2, double& coreE) {
  /*!
     Calculates the energy of the determinant.
   */
  double energy = 0.0;
  size_t one = 1;
  vector<int> closed;
  for (int i = 0; i < EffDetLen; i++) {
    long reprBit = repr[i];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      closed.push_back(i * 64 + pos - 1);
      reprBit &= ~(one << (pos - 1));
    }
  }

  for (int i = 0; i < (int)closed.size(); i++) {
    int I = closed.at(i);
    energy += I1(I, I);
    for (int j = i + 1; j < (int)closed.size(); j++) {
      int J = closed.at(j);
      energy += I2.Direct(I / 2, J / 2);
      if ((I % 2) == (J % 2)) {
        energy -= I2.Exchange(I / 2, J / 2);
      }
    }
  }

  return energy + coreE;
}

//=============================================================================
CItype Determinant::Hij_2Excite(int& i, int& j, int& a, int& b, oneInt& I1,
                                twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by
  a double excitation.
   */
  double sgn = 1.0;
  int I = min(i, j), J = max(i, j), A = min(a, b), B = max(a, b);
  parity(min(I, A), max(I, A), sgn);
  parity(min(J, B), max(J, B), sgn);
  if (A > J || B < I) sgn *= -1.;
  return sgn * (I2(A, I, B, J) - I2(A, J, B, I));
}

//=============================================================================
CItype Determinant::Hij_1Excite(int& a, int& i, oneInt& I1, twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by
  a single excitation a^dag_a a_i.
   */
  double sgn = 1.0;
  parity(min(a, i), max(a, i), sgn);

  CItype energy = I1(a, i);
  long one = 1;
  for (int I = 0; I < EffDetLen; I++) {
    long reprBit = repr[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I * 64 + pos - 1;
      energy += (I2(a, i, j, j) - I2(a, j, j, i));
      reprBit &= ~(one << (pos - 1));
    }
  }
  energy *= sgn;
  return energy;
}

//=============================================================================
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
           double& coreE, size_t& orbDiff) {
  /*!
  Calculates the hamiltonian matrix element connecting the two determinants bra
  and ket.
   */
  int cre[200], des[200], ncre = 0, ndes = 0;
  long u, b, k, one = 1;
  cre[0] = -1;
  cre[1] = -1;
  des[0] = -1;
  des[1] = -1;

  for (int i = 0; i < Determinant::EffDetLen; i++) {
    u = bra.repr[i] ^ ket.repr[i];
    b = u & bra.repr[i];  // the cre bits
    k = u & ket.repr[i];  // the des bits

    while (b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncre] = pos - 1 + i * 64;
      ncre++;
      b &= ~(one << (pos - 1));
    }
    while (k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndes] = pos - 1 + i * 64;
      ndes++;
      k &= ~(one << (pos - 1));
    }
  }

  if (ncre == 0) {
    cout << "Use the function for energy" << endl;
    exit(0);
  } else if (ncre == 1) {
    size_t c0 = cre[0], N = bra.norbs, d0 = des[0];
    orbDiff = c0 * N + d0;
    return ket.Hij_1Excite(cre[0], des[0], I1, I2);
  } else if (ncre == 2) {
    size_t c0 = cre[0], c1 = cre[1], d1 = des[1], N = bra.norbs, d0 = des[0];
    orbDiff = c1 * N * N * N + d1 * N * N + c0 * N + d0;
    return ket.Hij_2Excite(des[0], des[1], cre[0], cre[1], I1, I2);
  } else {
    return 0.;
  }
}
