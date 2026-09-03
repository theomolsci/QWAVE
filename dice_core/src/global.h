/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Developed by Sandeep Sharma with contributions from James E. T. Smith and
  Adam A. Holmes, 2017. Copyright (c) 2017, Sandeep Sharma.

  This trimmed copy removes MPI / boost::interprocess shared-memory globals;
  the code is always compiled serial (one process), with std::thread used for
  intra-process parallelism instead.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.
*/
#ifndef Global_HEADER_H
#define Global_HEADER_H
#include <string>

typedef unsigned short ushort;
// DetLen * 64 bits = number of spin orbitals supported.
// DetLen = 2 -> 128 spin orbitals -> up to 64 spatial orbitals.
const int DetLen = 2;
extern double startofCalc;
double getTime();

#define MatrixXx MatrixXd
#define CItype double

// Serial stand-ins for the MPI rank/size globals referenced by extracted code.
extern int commrank, shmrank, localrank;
extern int commsize, shmsize, localsize;
#endif
