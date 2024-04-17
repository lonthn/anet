//
// Created by zeqi.luo on 2021/1/27.
//

#include "include/layer/dropout_layer.h"

#include <random>

namespace anet {

DropoutLayer::DropoutLayer(float_t p)
: p_(p) {
}

void DropoutLayer::calcY(mat_t &x, mat_t &y) {
  uint64_t seed = 0;
  std::default_random_engine e(seed);
  std::bernoulli_distribution bd(1-p_);
  for (int32_t i = 0; i < y.size(); i++) {
    if (bd(e)) {
      y[i] = x[i] / (1-p_);
    }
  }
}

}