//
// Created by zeqi.luo on 2020/10/22.
//


#ifndef LIBOW_ACTIVATION_LAYER_H
#define LIBOW_ACTIVATION_LAYER_H

#include "include/layer/layer.h"
#include "include/math/functions.h"

namespace anet {

class ActivationLayer : public Layer {
public:
  enum Type {
    kReLU,
    kELU,
    kSigmoid,
    kTanH,
    kSoftmax,
  };
  ActivationLayer() = default;
  ActivationLayer(Type type);
  std::string name() override {
    return "activation";
  }

  void calcY(mat_t &x, mat_t &y) override;
  mat_t calcD(mat_t& x, mat_t& y, mat_t& d,
              std::vector<mat_t>& g) override;

  void writeTo(obstream& out) override;
  void readFrom(ibstream& in) override;

private:
  Type type_;
  float_t alpha_;
};

}

#endif //LIBOW_ACTIVATION_LAYER_H
