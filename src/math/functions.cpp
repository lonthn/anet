//
// Created by zeqi.luo on 2020/10/29.
//

#include "include/math/functions.h"

#include <utility>

float_t MSEFunc::f(mat_t& y, mat_t& a) {
  float_t sum = 0;
  for (int32_t i = 0; i < y.size(); i++) {
    sum += (y[i] - a[i]) * (y[i] - a[i]);
  }
  return sum / (float_t(2));
}

mat_t MSEFunc::d(mat_t& y, mat_t& a) {
  mat_t g(y.shape3());
  for (int32_t i = 0; i < y.size(); i++) {
    g[i] = a[i] - y[i];
  }
  return g;
}

float_t CrossEntropyFunc::f(mat_t& y, mat_t& a) {
  float_t sum = 0;
  for (int32_t i = 0; i < y.size(); i++) {
    if (a[i] != 0 && a[i] != 1) {
      sum += y[i] * std::log(a[i]) + (1.0 - y[i]) * std::log(1.0 - a[i]);
    }
  }
  return -sum;
}

mat_t CrossEntropyFunc::d(mat_t& y, mat_t& a) {
  return a - y;
}

float_t QuadraticCost::f(mat_t& y, mat_t& a) {
  mat_t temp = a - y;
  float_t num = 0;
  for (int32_t i = 0; i < temp.size(); i++) {
    num += temp[i] * temp[i];
  }
  return 0.5 * std::pow(std::sqrt(num), 2);
}