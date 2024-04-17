//
// Created by zeqi.luo on 2020/10/29.
//

#ifndef LIBOW_FUNCTIONS_H
#define LIBOW_FUNCTIONS_H

#include "include/math/matrix.h"
#include "include/types.h"

struct LossFunction {
  virtual float_t f(mat_t& y, mat_t& a) = 0;
  virtual mat_t d(mat_t& y, mat_t& a) = 0;
};

// mean squared error
struct MSEFunc : LossFunction {
  float_t f(mat_t& y, mat_t& a) override;
  mat_t d(mat_t& y, mat_t& a) override;
};

struct CrossEntropyFunc : LossFunction {
  float_t f(mat_t& y, mat_t& a) override;
  mat_t d(mat_t& y, mat_t& a) override;
};

struct QuadraticCost : LossFunction {
  float_t f(mat_t& y, mat_t& a) override;
  mat_t d(mat_t& y, mat_t& a) override {return mat_t(); }
};

#endif //LIBOW_FUNCTIONS_H
