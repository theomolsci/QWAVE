/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  See SHCImakeHamiltonian.h for what was trimmed.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#include "SHCImakeHamiltonian.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include "Determinants.h"
#include "integral.h"

using namespace std;
using namespace SHCISortMpiUtils;

//=============================================================================
// small sort/search utilities copied from Dice's SHCISortMpiUtils.cpp
//=============================================================================
int SHCISortMpiUtils::binarySearch(int* arr, int l, int r, int x) {
  if (r >= l) {
    int mid = l + (r - l) / 2;

    // If the element is present at the middle itself
    if (arr[mid] == x) return mid;

    // If element is smaller than mid, then it can only be present
    // in left subarray
    if (arr[mid] > x) return binarySearch(arr, l, mid - 1, x);

    // Else the element can only be present in right subarray
    return binarySearch(arr, mid + 1, r, x);
  }

  // We reach here when element is not present in array
  return -1;
}

void SHCISortMpiUtils::merge(int* a, long low, long high, long mid, int* x,
                             int* c, int* cx) {
  long i, j, k;
  i = low;
  k = low;
  j = mid + 1;
  while (i <= mid && j <= high) {
    if (a[i] < a[j]) {
      c[k] = a[i];
      cx[k] = x[i];
      k++;
      i++;
    } else {
      c[k] = a[j];
      cx[k] = x[j];
      k++;
      j++;
    }
  }
  while (i <= mid) {
    c[k] = a[i];
    cx[k] = x[i];
    k++;
    i++;
  }
  while (j <= high) {
    c[k] = a[j];
    cx[k] = x[j];
    k++;
    j++;
  }
  for (i = low; i < k; i++) {
    a[i] = c[i];
    x[i] = cx[i];
  }
}

void SHCISortMpiUtils::mergesort(int* a, long low, long high, int* x, int* c,
                                 int* cx) {
  long mid;
  if (low < high) {
    mid = (low + high) / 2;
    mergesort(a, low, mid, x, c, cx);
    mergesort(a, mid + 1, high, x, c, cx);
    merge(a, low, high, mid, x, c, cx);
  }
  return;
}

//=============================================================================
void SHCImakeHamiltonian::HamHelpers2::PopulateHelpers(Determinant* Dets,
                                                       int DetsSize,
                                                       int startIndex) {
  SHCImakeHamiltonian::PopulateHelperLists2(
      BetaN, AlphaN, AlphaMajorToBeta, AlphaMajorToDet, BetaMajorToAlpha,
      BetaMajorToDet, SinglesFromAlpha, SinglesFromBeta, Dets, DetsSize,
      startIndex);
}

//=============================================================================
static void updateAlphaBeta(HalfDet& da, HalfDet& db,
                            std::map<HalfDet, int>& BetaN,
                            std::map<HalfDet, int>& AlphaN,
                            vector<vector<int>>& AlphaMajorToBeta,
                            vector<vector<int>>& AlphaMajorToDet,
                            vector<vector<int>>& BetaMajorToAlpha,
                            vector<vector<int>>& BetaMajorToDet,
                            vector<vector<int>>& SinglesFromAlpha,
                            vector<vector<int>>& SinglesFromBeta,
                            int DetIndex) {
  // number of SPATIAL orbitals actually in play (Dice used the full
  // 64*DetLen/2 capacity here; restricting to the real orbital count is a
  // pure optimization, single excitations to orbitals >= nSpatOrbs can never
  // be found in the maps).
  int nSpatOrbs = Determinant::norbs / 2;
  int norbs = 64 * DetLen;

  auto itb = BetaN.find(db);
  if (itb == BetaN.end()) {
    auto ret = BetaN.insert(std::pair<HalfDet, int>(db, BetaMajorToDet.size()));
    itb = ret.first;
    BetaMajorToAlpha.resize(itb->second + 1);
    BetaMajorToDet.resize(itb->second + 1);
    SinglesFromBeta.resize(itb->second + 1);

    std::vector<int> closedb(norbs / 2);
    std::vector<int> openb(norbs / 2, 0);
    int nclosedb = db.getOpenClosed(openb, closedb);

    for (int j = 0; j < nclosedb; j++)
      for (int k = 0; k < norbs / 2 - nclosedb; k++) {
        if (openb[k] >= nSpatOrbs) break;
        HalfDet dbcopy = db;
        dbcopy.setocc(closedb[j], false);
        dbcopy.setocc(openb[k], true);
        auto itbcopy = BetaN.find(dbcopy);
        if (itbcopy != BetaN.end()) {
          SinglesFromBeta[itb->second].push_back(itbcopy->second);
          SinglesFromBeta[itbcopy->second].push_back(itb->second);
        }
      }
  }  // itb

  auto ita = AlphaN.find(da);
  if (ita == AlphaN.end()) {
    auto ret =
        AlphaN.insert(std::pair<HalfDet, int>(da, AlphaMajorToDet.size()));
    ita = ret.first;
    AlphaMajorToBeta.resize(ita->second + 1);
    AlphaMajorToDet.resize(ita->second + 1);
    SinglesFromAlpha.resize(ita->second + 1);

    std::vector<int> closeda(norbs / 2);
    std::vector<int> opena(norbs / 2, 0);
    int ncloseda = da.getOpenClosed(opena, closeda);

    for (int j = 0; j < ncloseda; j++)
      for (int k = 0; k < norbs / 2 - ncloseda; k++) {
        if (opena[k] >= nSpatOrbs) break;
        HalfDet dacopy = da;
        dacopy.setocc(closeda[j], false);
        dacopy.setocc(opena[k], true);
        auto itacopy = AlphaN.find(dacopy);
        if (itacopy != AlphaN.end()) {
          SinglesFromAlpha[ita->second].push_back(itacopy->second);
          SinglesFromAlpha[itacopy->second].push_back(ita->second);
        }
      }
  }  // ita

  AlphaMajorToBeta[ita->second].push_back(itb->second);
  AlphaMajorToDet[ita->second].push_back(DetIndex);

  BetaMajorToAlpha[itb->second].push_back(ita->second);
  BetaMajorToDet[itb->second].push_back(DetIndex);
}  // end updateAlphaBeta

//=============================================================================
void SHCImakeHamiltonian::PopulateHelperLists2(
    std::map<HalfDet, int>& BetaN, std::map<HalfDet, int>& AlphaN,
    vector<vector<int>>& AlphaMajorToBeta, vector<vector<int>>& AlphaMajorToDet,
    vector<vector<int>>& BetaMajorToAlpha, vector<vector<int>>& BetaMajorToDet,
    vector<vector<int>>& SinglesFromAlpha, vector<vector<int>>& SinglesFromBeta,
    Determinant* Dets, int DetsSize, int StartIndex) {
  /*!
  ith vector of AlphaMajor contains all Determinants that have ith Alpha
  string; AlphaMajorToBeta/AlphaMajorToDet give the beta-string index and the
  (1-based) determinant index, respectively.
  */
  for (int i = StartIndex; i < DetsSize; i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();
    updateAlphaBeta(da, db, BetaN, AlphaN, AlphaMajorToBeta, AlphaMajorToDet,
                    BetaMajorToAlpha, BetaMajorToDet, SinglesFromAlpha,
                    SinglesFromBeta, i + 1);
    // Time-reversal branch removed (Trev is always 0 here).
  }

  for (size_t i = 0; i < AlphaMajorToBeta.size(); i++) {
    vector<int> betacopy = AlphaMajorToBeta[i];
    vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
    for (size_t j = 0; j < detIndex.size(); j++) detIndex[j] = j;
    mergesort(&betacopy[0], 0, betacopy.size() - 1, &detIndex[0],
              &AlphaMajorToBeta[i][0], &detIndexCopy[0]);
    detIndexCopy.clear();
    reorder(AlphaMajorToDet[i], detIndex);

    std::sort(SinglesFromAlpha[i].begin(), SinglesFromAlpha[i].end());
  }

  for (size_t i = 0; i < BetaMajorToAlpha.size(); i++) {
    vector<int> betacopy = BetaMajorToAlpha[i];
    vector<int> detIndex(betacopy.size(), 0), detIndexCopy(betacopy.size(), 0);
    for (size_t j = 0; j < detIndex.size(); j++) detIndex[j] = j;
    mergesort(&betacopy[0], 0, betacopy.size() - 1, &detIndex[0],
              &BetaMajorToAlpha[i][0], &detIndexCopy[0]);
    detIndexCopy.clear();
    reorder(BetaMajorToDet[i], detIndex);

    std::sort(SinglesFromBeta[i].begin(), SinglesFromBeta[i].end());
  }
}  // end SHCImakeHamiltonian::PopulateHelperLists2

//=============================================================================
void SHCImakeHamiltonian::SparseHam::makeFromHelper(HamHelpers2& helper2,
                                                    Determinant* Dets,
                                                    int DetsSize, oneInt& I1,
                                                    twoInt& I2, double& coreE,
                                                    int nThreads) {
  /*!
  Dice's MakeHfromSMHelpers2, serial path (nprocs=1, offSet=0, Trev=0, no
  RDM), with the alpha-string loop distributed over std::threads.  Each
  determinant I appears in exactly one (alpha-string i, slot ii), so row I of
  connections/Helements is written by exactly one thread and the resulting
  matrix is bit-for-bit independent of nThreads.
  */
  if (nThreads < 1) nThreads = 1;

  vector<vector<int>>& AlphaMajorToBeta = helper2.AlphaMajorToBeta;
  vector<vector<int>>& AlphaMajorToDet = helper2.AlphaMajorToDet;
  vector<vector<int>>& BetaMajorToAlpha = helper2.BetaMajorToAlpha;
  vector<vector<int>>& BetaMajorToDet = helper2.BetaMajorToDet;
  vector<vector<int>>& SinglesFromAlpha = helper2.SinglesFromAlpha;
  vector<vector<int>>& SinglesFromBeta = helper2.SinglesFromBeta;

  connections.clear();
  Helements.clear();
  connections.resize(DetsSize);
  Helements.resize(DetsSize);

  // diagonal elements (each row starts with its diagonal, as in Dice)
  {
    std::atomic<int> nextRow(0);
    auto diagWork = [&]() {
      int k;
      while ((k = nextRow.fetch_add(512)) < DetsSize) {
        int kend = std::min(k + 512, DetsSize);
        for (; k < kend; k++) {
          connections[k].push_back(k);
          CItype hij = Dets[k].Energy(I1, I2, coreE);
          Helements[k].push_back(hij);
        }
      }
    };
    vector<std::thread> threads;
    for (int t = 1; t < nThreads; t++) threads.emplace_back(diagWork);
    diagWork();
    for (auto& th : threads) th.join();
  }

  // off-diagonal elements: distribute alpha strings over threads
  std::atomic<size_t> nextAlpha(0);
  size_t nAlpha = AlphaMajorToBeta.size();

  auto offDiagWork = [&]() {
    size_t i;
    while ((i = nextAlpha.fetch_add(1)) < nAlpha) {
      for (size_t ii = 0; ii < AlphaMajorToBeta[i].size(); ii++) {
        int Astring = i, Bstring = AlphaMajorToBeta[i][ii],
            DetI = AlphaMajorToDet[i][ii];
        int row = std::abs(DetI) - 1;

        // singles from Astring
        for (size_t j = 0; j < SinglesFromAlpha[Astring].size(); j++) {
          int Asingle = SinglesFromAlpha[Astring][j];
          int index = binarySearch(&BetaMajorToAlpha[Bstring][0], 0,
                                   BetaMajorToAlpha[Bstring].size() - 1,
                                   Asingle);
          if (index != -1) {
            int DetJ = BetaMajorToDet[Bstring][index];
            if (std::abs(DetJ) >= std::abs(DetI)) continue;
            size_t orbDiff;
            CItype hij = Hij(Dets[std::abs(DetJ) - 1], Dets[std::abs(DetI) - 1],
                             I1, I2, coreE, orbDiff);
            connections[row].push_back(std::abs(DetJ) - 1);
            Helements[row].push_back(hij);
          }
        }

        // single Alpha and single Beta
        for (size_t j = 0; j < SinglesFromAlpha[Astring].size(); j++) {
          int Asingle = SinglesFromAlpha[Astring][j];

          int SearchStartIndex = 0,
              AlphaToBetaLen = AlphaMajorToBeta[Asingle].size(),
              SinglesFromBLen = SinglesFromBeta[Bstring].size();
          for (int k = 0; k < SinglesFromBLen; k++) {
            int& Bsingle = SinglesFromBeta[Bstring][k];

            if (SearchStartIndex >= AlphaToBetaLen) break;

            int index = SearchStartIndex;
            for (; index < AlphaToBetaLen &&
                   AlphaMajorToBeta[Asingle][index] < Bsingle;
                 index++) {
            }

            SearchStartIndex = index;
            if (index < AlphaToBetaLen &&
                AlphaMajorToBeta[Asingle][index] == Bsingle) {
              int DetJ = AlphaMajorToDet[Asingle][SearchStartIndex];
              if (std::abs(DetJ) >= std::abs(DetI)) continue;
              size_t orbDiff;
              CItype hij = Hij(Dets[std::abs(DetJ) - 1],
                               Dets[std::abs(DetI) - 1], I1, I2, coreE,
                               orbDiff);
              connections[row].push_back(std::abs(DetJ) - 1);
              Helements[row].push_back(hij);
            }  //*itb == Bsingle
          }    // k 0->SinglesFromBeta
        }      // j singles fromAlpha

        // singles from Bstring
        for (size_t j = 0; j < SinglesFromBeta[Bstring].size(); j++) {
          int Bsingle = SinglesFromBeta[Bstring][j];

          int index = binarySearch(&AlphaMajorToBeta[Astring][0], 0,
                                   AlphaMajorToBeta[Astring].size() - 1,
                                   Bsingle);

          if (index != -1) {
            int DetJ = AlphaMajorToDet[Astring][index];
            if (std::abs(DetJ) >= std::abs(DetI)) continue;

            size_t orbDiff;
            CItype hij = Hij(Dets[std::abs(DetJ) - 1], Dets[std::abs(DetI) - 1],
                             I1, I2, coreE, orbDiff);
            connections[row].push_back(std::abs(DetJ) - 1);
            Helements[row].push_back(hij);
          }
        }

        // double beta excitation
        for (size_t j = 0; j < AlphaMajorToBeta[i].size(); j++) {
          int DetJ = AlphaMajorToDet[i][j];
          if (std::abs(DetJ) >= std::abs(DetI)) continue;

          Determinant di = Dets[std::abs(DetI) - 1];
          if (Dets[std::abs(DetJ) - 1].ExcitationDistance(di) == 2) {
            size_t orbDiff;
            CItype hij = Hij(Dets[std::abs(DetJ) - 1], Dets[std::abs(DetI) - 1],
                             I1, I2, coreE, orbDiff);
            connections[row].push_back(std::abs(DetJ) - 1);
            Helements[row].push_back(hij);
          }
        }

        // double Alpha excitation
        for (size_t j = 0; j < BetaMajorToAlpha[Bstring].size(); j++) {
          int DetJ = BetaMajorToDet[Bstring][j];
          if (std::abs(DetJ) >= std::abs(DetI)) continue;

          Determinant dj = Dets[std::abs(DetI) - 1];
          if (Dets[std::abs(DetJ) - 1].ExcitationDistance(dj) == 2) {
            size_t orbDiff;
            CItype hij = Hij(Dets[std::abs(DetJ) - 1], Dets[std::abs(DetI) - 1],
                             I1, I2, coreE, orbDiff);
            connections[row].push_back(std::abs(DetJ) - 1);
            Helements[row].push_back(hij);
          }
        }
      }  // ii
    }    // i
  };

  vector<std::thread> threads;
  for (int t = 1; t < nThreads; t++) threads.emplace_back(offDiagWork);
  offDiagWork();
  for (auto& th : threads) th.join();
}  // end SHCImakeHamiltonian::SparseHam::makeFromHelper
