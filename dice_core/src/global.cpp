/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
  Serial stubs for the globals declared in global.h.
*/
#include "global.h"
#include <chrono>

double getTime() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

double startofCalc = getTime();

int commrank = 0, shmrank = 0, localrank = 0;
int commsize = 1, shmsize = 1, localsize = 1;
