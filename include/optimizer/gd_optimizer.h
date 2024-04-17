//
// Created by zeqi.luo on 2021/1/14.
//

#ifndef ANET_GD_OPTIMIZER_H
#define ANET_GD_OPTIMIZER_H

#include "include/optimizer/optimizer.h"

namespace anet {

class GDOptimizer : public Optimizer {
public:
  explicit GDOptimizer(float_t alpha = 0.01)
    : alpha_(alpha) { }

  void operator () (vector<mat_t>& var,
                    vector<mat_t>& grad,
                    float_t n) override {
    int32_t size = var.size();
    for (int32_t i = 0; i < size; i++) {
      int32_t varn = var[i].size();
      float_t* varptr = var[i].data();
      float_t* gradptr = grad[i].data();
      for (int32_t j = 0; j < varn; j++) {
        varptr[j] -= alpha_ * gradptr[j] / n;
      }
    }
  }

private:
  float_t alpha_;
};

}

#endif //ANET_GD_OPTIMIZER_H
