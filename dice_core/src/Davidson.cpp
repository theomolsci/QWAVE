/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  This is Dice's davidson() (the SERIAL code path), with:
    - AllocateSHM (boost::interprocess shared memory) replaced by plain heap
      buffers,
    - all MPI broadcast/reduce calls compiled out (single process),
    - boost::format prints replaced by printf,
    - the hard-coded iteration caps (800*nroots soft / 2000*nroots exit(0))
      replaced by a caller-supplied maxIter: on hitting it, a warning is
      printed and the current best estimate is returned instead of aborting.
  The subspace iteration, Rayleigh-Ritz, preconditioning, deflation and
  restart logic are otherwise unchanged.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#include "Davidson.h"
#include <Eigen/Dense>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "Hmult.h"
#include "global.h"

using namespace Eigen;
using namespace std;

//=============================================================================
void precondition(MatrixXx& r, MatrixXx& diag, double& e) {
  /*!
  Properly precondition the matrix "r"
  */
  for (int i = 0; i < r.rows(); i++) {
    if (abs(e - diag(i, 0)) > 1e-12)
      r(i, 0) = r(i, 0) / (e - diag(i, 0));
    else
      r(i, 0) = r(i, 0) / (e - diag(i, 0) - 1.e-12);
  }
}  // end precondition

//=============================================================================
vector<double> davidson(Hmult2& H, vector<MatrixXx>& x0, MatrixXx& diag,
                        int maxCopies, double tol, int& numIter, int maxIter,
                        bool print) {
  std::vector<double> eroots;

  int nroots = x0.size();
  int brows = x0[0].rows();

  std::vector<CItype> bcolVec(brows, 0.0), sigmacolVec(brows, 0.0);
  CItype* bcol = &bcolVec[0];
  CItype* sigmacol = &sigmacolVec[0];

  MatrixXx b = MatrixXx::Zero(brows, maxCopies);

  // if some vector has zero norm then randomise it
  for (int i = 0; i < nroots; i++) {
    b.col(i) = 1. * x0[i];
    if (x0[i].norm() < 1.e-10) {
      b.col(i).setRandom();
      b.col(i) = b.col(i) / b.col(i).norm();
    }
  }

  // make vectors orthogonal to each other
  for (int i = 0; i < (int)x0.size(); i++) {
    for (int j = 0; j < i; j++) {
      CItype overlap = (b.col(j).adjoint() * b.col(i))(0, 0);
      b.col(i) -= overlap * b.col(j);
    }
    if (b.col(i).norm() < 1e-8) {
      b.col(i).setRandom();
    }
    for (int j = 0; j < i; j++) {
      CItype overlap = (b.col(j).adjoint() * b.col(i))(0, 0);
      b.col(i) -= overlap * b.col(j);
    }
    b.col(i) = b.col(i) / b.col(i).norm();
  }  // i

  MatrixXx sigma = MatrixXx::Zero(brows, maxCopies);

  int sigmaSize = 0, bsize = x0.size();
  MatrixXx r = MatrixXx::Zero(brows, 1);
  int convergedRoot = 0;

  numIter = 0;
  double ei = 0.0;
  while (true) {
    // 0->continue with the loop, 2 -> return
    int continueOrReturn = 0;

    for (int i = sigmaSize; i < bsize; i++) {
      for (int k = 0; k < brows; k++) bcol[k] = b(k, i);
      for (int k = 0; k < brows; k++) sigmacol[k] = 0.0;

      H(bcol, sigmacol);
      sigmaSize++;

      for (int k = 0; k < brows; k++) sigma(k, i) = sigmacol[k];
    }  // i

    {
      MatrixXx hsubspace(bsize, bsize);
      hsubspace.setZero(bsize, bsize);
      for (int i = 0; i < bsize; i++)
        for (int j = i; j < bsize; j++) {
          hsubspace(i, j) = b.col(i).dot(sigma.col(j));
          hsubspace(j, i) = hsubspace(i, j);
        }
      SelfAdjointEigenSolver<MatrixXx> eigensolver(hsubspace);
      if (eigensolver.info() != Success) {
        throw std::runtime_error("davidson: subspace eigenvalue solver failed");
      }

      b.block(0, 0, b.rows(), bsize) =
          b.block(0, 0, b.rows(), bsize) * eigensolver.eigenvectors();
      sigma.block(0, 0, b.rows(), bsize) =
          sigma.block(0, 0, b.rows(), bsize) * eigensolver.eigenvectors();

      ei = eigensolver.eigenvalues()[convergedRoot];
      for (int i = 0; i < convergedRoot; i++) {
        r = sigma.col(i) - eigensolver.eigenvalues()[i] * b.col(i);
        double error = r.norm();
        if (error > tol) {
          convergedRoot = i;
          if (print) printf("going back to converged root %d\n", i);
          continue;
        }
      }

      r = sigma.col(convergedRoot) - ei * b.col(convergedRoot);
      double error = r.norm();
      if (print) {
        if (numIter == 0)
          printf("nIter  Root               Energy                Error\n");
        printf("%5i  %4i   %18.10g   %18.10g  %10.2f\n", numIter,
               convergedRoot, ei, error, (getTime() - startofCalc));
      }
      numIter++;

      if (hsubspace.rows() == b.rows()) {
        // all roots are available
        for (int i = 0; i < (int)x0.size(); i++) {
          x0[i] = b.col(i);
          eroots.push_back(eigensolver.eigenvalues()[i]);
          if (print)
            printf("#niter:%3d root:%3d -> Energy : %18.10g\n", numIter, i,
                   eroots[i]);
        }
        continueOrReturn = 2;
        goto label1;
      }

      if (error < tol || numIter >= maxIter) {
        if (error >= tol && print)
          printf(
              "davidson: max_iter %d reached for root %d, residual %g > tol "
              "%g; returning current estimate\n",
              maxIter, convergedRoot, error, tol);
        convergedRoot++;
        if (print)
          printf("#niter:%3d root:%3d -> Energy : %18.10g\n", numIter,
                 convergedRoot - 1, ei);
        if (convergedRoot == nroots) {
          for (int i = 0; i < convergedRoot; i++) {
            x0[i] = b.col(i);
            eroots.push_back(eigensolver.eigenvalues()[i]);
          }
          continueOrReturn = 2;
          goto label1;
        }
      }  // cvg
    }

  label1:
    if (continueOrReturn == 2) return eroots;

    {
      precondition(r, diag, ei);
      for (int i = 0; i < bsize; i++)
        r = r - (b.col(i).adjoint() * r)(0, 0) * b.col(i) /
                    (b.col(i).adjoint() * b.col(i));
      if (r.norm() < 1e-10) {
        // if preconditioned r is in the subspace
        // try adding random vectors
        bool success = false;
        for (int attempt = 0; attempt < 3 && !success; attempt++) {
          r.setRandom();
          for (int i = 0; i < bsize; i++)
            r = r - (b.col(i).adjoint() * r)(0, 0) * b.col(i) /
                        (b.col(i).adjoint() * b.col(i));
          if (r.norm() > 1e-10) success = true;
        }
        if (success) {
          if (bsize < maxCopies) {
            b.col(bsize) = r / r.norm();
            bsize++;
          } else {
            bsize = nroots + 3;
            sigmaSize = nroots + 2;
            b.col(bsize - 1) = r / r.norm();
          }
        } else {
          throw std::runtime_error(
              "davidson: BREAKDOWN, could not generate a new subspace vector");
        }
      } else {
        if (bsize < maxCopies) {
          b.col(bsize) = r / r.norm();
          bsize++;
        } else {
          bsize = nroots + 3;
          sigmaSize = nroots + 2;
          b.col(bsize - 1) = r / r.norm();
        }
      }
    }
  }  // while
}  // end davidson
