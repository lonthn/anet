//
// Created by zeqi.luo on 2020/12/11.
//

#ifndef ANET_OPTIMIZER_H
#define ANET_OPTIMIZER_H

#include "include/types.h"
#include "include/math/matrix.h"
#include "include/network.h"

#include <map>
#include <vector>

namespace anet {

using std::vector;

class Optimizer {
public:
  static void initialOnce(vector<mat_t>& param, vector<mat_t>& src) {
    if (param.size() != src.size()) {
      param.reserve(src.size());
      for (auto &o : src) {
        param.emplace_back(o.shape3());
      }
    }
  }

  virtual void operator () (vector<mat_t>& var,
                            vector<mat_t>& grad,
                            float_t n) = 0;
};

}

#endif //ANET_OPTIMIZER_H
