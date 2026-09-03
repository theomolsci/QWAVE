/*
  Extracted from Dice (https://github.com/caleb-johnson/Dice), GPLv3.
*/
#include "integral.h"

// thread_local so concurrent read accesses to mixed-spin elements from the
// std::thread workers are race-free (Dice used a plain member, which was fine
// in its single-threaded-per-process MPI model).
thread_local double twoInt::zero = 0.0;
