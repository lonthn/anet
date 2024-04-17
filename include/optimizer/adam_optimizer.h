//
// Created by zeqi.luo on 2020/12/11.
//


#ifndef LIBOW_ADAMOPTIMIZER_H
#define LIBOW_ADAMOPTIMIZER_H

#include "include/optimizer/optimizer.h"

namespace anet {

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float_t alpha, float_t b1, float_t b2, float_t epsilon)
  : alpha_(alpha)
  , beta1_(b1)
  , beta2_(b2)
  , beta1Power_(b1)
  , beta2Power_(b2)
  , epsilon_(epsilon) { }

  void operator () (vector<mat_t>& var,
                    vector<mat_t>& grad,
                    float_t n) override {
    Optimizer::initialOnce(m_, var);
    Optimizer::initialOnce(v_, var);
    for_ij(var.size(), var[i].size()) {
      float_t g = grad[i][j] / n;
      m_[i][j] = m_[i][j] * beta1_ + g * (1 - beta1_);
      v_[i][j] = v_[i][j] * beta2_ + g*g * (1 - beta2_);
      float_t mm = m_[i][j] / (1 - beta1Power_);
      float_t vv = v_[i][j] / (1 - beta2Power_);
      var[i][j] -= alpha_ / (std::sqrt(vv) + epsilon_) * mm;
    }
    beta1Power_ *= beta1_;
    beta2Power_ *= beta2_;
  }

private:
  float_t alpha_;
  float_t beta1_;
  float_t beta2_;
  float_t beta1Power_;
  float_t beta2Power_;
  float_t epsilon_;

  std::vector<mat_t> m_;
  std::vector<mat_t> v_;
};

}

#endif //LIBOW_ADAMOPTIMIZER_H