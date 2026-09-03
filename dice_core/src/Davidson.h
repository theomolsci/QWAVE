/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "global.h"

struct Hmult2;
using namespace Eigen;

void precondition(MatrixXx& r, MatrixXx& diag, double& e);

// Dice's davidson(); maxIter added (Dice hard-coded 800*nroots) -- if the
// iteration cap is hit before the residual drops below tol a warning is
// printed and the current best estimate is returned.
std::vector<double> davidson(Hmult2& H, std::vector<MatrixXx>& x0,
                             MatrixXx& diag, int maxCopies, double tol,
                             int& numIter, int maxIter, bool print);

#endif
