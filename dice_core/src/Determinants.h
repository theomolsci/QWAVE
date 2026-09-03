/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  Trimmed for the fixed-basis solver: boost serialization, lexical-order
  hashing, time-reversal helpers and SOC support removed.  The bit-packing,
  parity and Slater-Condon matrix-element code is Dice's, unmodified.

  Spin-orbital bit convention (verified against Dice's FCIDUMP reader and
  Nalpha()/getAlpha()): bit 2*i   = alpha spatial orbital i
                        bit 2*i+1 = beta  spatial orbital i
  Bit k lives at repr[k/64], position k%64 (LSB first).

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef Determinants_HEADER_H
#define Determinants_HEADER_H

#include <iostream>
#include <vector>
#include "global.h"

class oneInt;
class twoInt;

using namespace std;

inline int BitCount(long x) {
  x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return (x * 0x0101010101010101ULL) >> 56;
}

// This is used to store just the alpha or the beta sub string of the entire
// determinant
class HalfDet {
 public:
  long repr[DetLen / 2];
  static int norbs;
  HalfDet() {
    for (int i = 0; i < DetLen / 2; i++) repr[i] = 0;
  }

  // the comparison between determinants is performed
  bool operator<(const HalfDet& d) const {
    for (int i = DetLen / 2 - 1; i >= 0; i--) {
      if (repr[i] < d.repr[i])
        return true;
      else if (repr[i] > d.repr[i])
        return false;
    }
    return false;
  }

  bool operator==(const HalfDet& d) const {
    for (int i = DetLen / 2 - 1; i >= 0; i--)
      if (repr[i] != d.repr[i]) return false;
    return true;
  }

  int ExcitationDistance(const HalfDet& d) const {
    int ndiff = 0;
    for (int i = 0; i < DetLen / 2; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff / 2;
  }

  // set the occupation of the ith orbital
  void setocc(int i, bool occ) {
    long Integer = i / 64, bit = i % 64, one = 1;
    if (occ)
      repr[Integer] |= one << bit;
    else
      repr[Integer] &= ~(one << bit);
  }

  // get the occupation of the ith orbital
  bool getocc(int i) const {
    long Integer = i / 64, bit = i % 64, reprBit = repr[Integer];
    if ((reprBit >> bit & 1) == 0)
      return false;
    else
      return true;
  }

  int getClosed(vector<int>& closed) {
    int cindex = 0;
    for (int i = 0; i < 32 * DetLen; i++) {
      if (getocc(i)) {
        closed.at(cindex) = i;
        cindex++;
      }
    }
    return cindex;
  }

  int getOpenClosed(vector<int>& open, vector<int>& closed) {
    int cindex = 0;
    int oindex = 0;
    for (int i = 0; i < 32 * DetLen; i++) {
      if (getocc(i)) {
        closed.at(cindex) = i;
        cindex++;
      } else {
        open.at(oindex) = i;
        oindex++;
      }
    }
    return cindex;
  }
};

class Determinant {
 public:
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long repr[DetLen];
  static char Trev;
  static int norbs;
  static int EffDetLen;

  Determinant() {
    for (int i = 0; i < DetLen; i++) repr[i] = 0;
  }

  Determinant(const Determinant& d) {
    for (int i = 0; i < DetLen; i++) repr[i] = d.repr[i];
  }

  void operator=(const Determinant& d) {
    for (int i = 0; i < DetLen; i++) repr[i] = d.repr[i];
  }

  double Energy(oneInt& I1, twoInt& I2, double& coreE);

  void parity(const int& start, const int& end, double& parity) {
    long one = 1;
    long mask = (one << (start % 64)) - one;
    long result = repr[start / 64] & mask;
    int nonZeroBits = -BitCount(result);

    for (int i = start / 64; i < end / 64; i++) {
      nonZeroBits += BitCount(repr[i]);
    }
    mask = (one << (end % 64)) - one;

    result = repr[end / 64] & mask;
    nonZeroBits += BitCount(result);

    parity *= (-2. * (nonZeroBits % 2) + 1);
    if (getocc(start)) parity *= -1.;

    return;
  }

  CItype Hij_1Excite(int& i, int& a, oneInt& I1, twoInt& I2);

  CItype Hij_2Excite(int& i, int& j, int& a, int& b, oneInt& I1, twoInt& I2);

  void flipAlphaBeta() {
    unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
    for (int i = 0; i < EffDetLen; i++)
      repr[i] = ((repr[i] & even) << 1) + ((repr[i] & odd) >> 1);
  }

  int Noccupied() const {
    int nelec = 0;
    for (int i = 0; i < EffDetLen; i++) {
      nelec += BitCount(repr[i]);
    }
    return nelec;
  }

  int Nalpha() const {
    int nalpha = 0;
    long alleven = 0x5555555555555555;
    for (int i = 0; i < EffDetLen; i++) {
      long even = repr[i] & alleven;
      nalpha += BitCount(even);
    }
    return nalpha;
  }

  int Nbeta() const {
    int nbeta = 0;
    long allodd = 0xAAAAAAAAAAAAAAAA;
    for (int i = 0; i < EffDetLen; i++) {
      long odd = repr[i] & allodd;
      nbeta += BitCount(odd);
    }
    return nbeta;
  }

  // Is the excitation between *this and d less than equal to 2.
  bool connected(const Determinant& d) const {
    int ndiff = 0;

    for (int i = 0; i < EffDetLen; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff <= 4;
  }

  // Get the number of electrons that need to be excited to get determinant d
  // from *this determinant e.g. single excitation will return 1
  int ExcitationDistance(const Determinant& d) const {
    int ndiff = 0;
    for (int i = 0; i < EffDetLen; i++) {
      ndiff += BitCount(repr[i] ^ d.repr[i]);
    }
    return ndiff / 2;
  }

  // Get HalfDet with just the alpha string
  HalfDet getAlpha() const {
    HalfDet d;
    for (int i = 0; i < EffDetLen; i++)
      for (int j = 0; j < 32; j++) {
        d.setocc(i * 32 + j, getocc(i * 64 + j * 2));
      }
    return d;
  }

  // get HalfDet with just the beta string
  HalfDet getBeta() const {
    HalfDet d;
    for (int i = 0; i < EffDetLen; i++)
      for (int j = 0; j < 32; j++)
        d.setocc(i * 32 + j, getocc(i * 64 + j * 2 + 1));
    return d;
  }

  // the comparison between determinants is performed
  bool operator<(const Determinant& d) const {
    for (int i = EffDetLen - 1; i >= 0; i--) {
      if (repr[i] < d.repr[i])
        return true;
      else if (repr[i] > d.repr[i])
        return false;
    }
    return false;
  }

  // check if the determinants are equal
  bool operator==(const Determinant& d) const {
    for (int i = EffDetLen - 1; i >= 0; i--)
      if (repr[i] != d.repr[i]) return false;
    return true;
  }

  // set the occupation of the ith orbital
  void setocc(int i, bool occ) {
    long Integer = i / 64, bit = i % 64, one = 1;
    if (occ)
      repr[Integer] |= one << bit;
    else
      repr[Integer] &= ~(one << bit);
  }

  // get the occupation of the ith orbital
  bool getocc(int i) const {
    long Integer = i / 64, bit = i % 64, reprBit = repr[Integer];
    if ((reprBit >> bit & 1) == 0)
      return false;
    else
      return true;
  }

  // returns integer array containing the closed and open orbital indices
  int getOpenClosed(int* open, int* closed) const {
    int oindex = 0, cindex = 0;
    for (int i = 0; i < norbs; i++) {
      if (getocc(i)) {
        closed[cindex] = i;
        cindex++;
      } else {
        open[oindex] = i;
        oindex++;
      }
    }
    return cindex;
  }
};

CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
           double& coreE, size_t& orbDiff);

#endif
