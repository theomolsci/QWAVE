/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  Trimmed: in-memory (heap) helper path only.  The boost::interprocess
  shared-memory segments, the MPI rank striping, the disk-batched SparseHam
  (writeBatch/readBatch via boost::serialization), time-reversal symmetry and
  the RDM orbDifference bookkeeping are removed.  The half-string grouping
  algorithm (PopulateHelperLists2) and the connection enumeration in
  makeFromHelper are Dice's MakeHfromSMHelpers2 logic, with the row loop
  parallelized over alpha strings with std::thread (each determinant appears
  in exactly one (alpha-string, ii) slot, so rows are written race-free).

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef SHCI_MAKEHAMILTONIAN_H
#define SHCI_MAKEHAMILTONIAN_H
#include <map>
#include <vector>
#include "global.h"

using namespace std;
class Determinant;
class HalfDet;
class oneInt;
class twoInt;

namespace SHCISortMpiUtils {
int binarySearch(int* arr, int l, int r, int x);
void merge(int* a, long low, long high, long mid, int* x, int* c, int* cx);
void mergesort(int* a, long low, long high, int* x, int* c, int* cx);
}  // namespace SHCISortMpiUtils

template <class T>
void reorder(vector<T>& A, std::vector<int>& reorder) {
  vector<T> Acopy = A;
  for (size_t i = 0; i < Acopy.size(); i++) {
    A[i] = Acopy[reorder[i]];
  }
}

namespace SHCImakeHamiltonian {

struct HamHelpers2 {
  vector<vector<int>> AlphaMajorToBeta;
  vector<vector<int>> AlphaMajorToDet;
  vector<vector<int>> BetaMajorToAlpha;
  vector<vector<int>> BetaMajorToDet;
  vector<vector<int>> SinglesFromAlpha;
  vector<vector<int>> SinglesFromBeta;
  map<HalfDet, int> BetaN;
  map<HalfDet, int> AlphaN;

  // routines
  void PopulateHelpers(Determinant* Dets, int DetsSize, int startIndex);

  void clear() {
    AlphaMajorToBeta.clear();
    AlphaMajorToDet.clear();
    BetaMajorToAlpha.clear();
    BetaMajorToDet.clear();
    SinglesFromAlpha.clear();
    SinglesFromBeta.clear();
    BetaN.clear();
    AlphaN.clear();
  }
};  // HamHelpers2

struct SparseHam {
  std::vector<std::vector<int>> connections;   // int32 column indices
  std::vector<std::vector<CItype>> Helements;  // float64 values
  SparseHam() {}

  void clear() {
    connections.clear();
    Helements.clear();
  }

  void resize(int size) {
    connections.resize(size);
    Helements.resize(size);
  }

  // Build the lower-triangle (J <= I) sparse Hamiltonian, row I owned by the
  // (alpha-string, ii) slot of determinant I; parallelized with nThreads
  // std::threads.
  void makeFromHelper(HamHelpers2& helper2, Determinant* Dets, int DetsSize,
                      oneInt& I1, twoInt& I2, double& coreE, int nThreads);
};  // SparseHam

void PopulateHelperLists2(
    std::map<HalfDet, int>& BetaN, std::map<HalfDet, int>& AlphaN,
    vector<vector<int>>& AlphaMajorToBeta, vector<vector<int>>& AlphaMajorToDet,
    vector<vector<int>>& BetaMajorToAlpha, vector<vector<int>>& BetaMajorToDet,
    vector<vector<int>>& SinglesFromAlpha, vector<vector<int>>& SinglesFromBeta,
    Determinant* Dets, int DetsSize, int StartIndex);

}  // namespace SHCImakeHamiltonian

#endif
