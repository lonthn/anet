//
// Created by zeqi.luo on 2020/10/23.
//

#ifndef LIBOW_RESHAPE_LAYER_H
#define LIBOW_RESHAPE_LAYER_H

#include "include/layer/layer.h"

namespace anet {

class ReshapeLayer : public Layer {
public:
  ReshapeLayer() = default;
  explicit ReshapeLayer(vec3_t shape);

  std::string name() override {
    return "reshape";
  }

  void calcY(mat_t &x, mat_t &y) override;
  mat_t calcD(mat_t& x, mat_t& y, mat_t& d, std::vector<mat_t>& g) override;

  void writeTo(obstream& out) override;
  void readFrom(ibstream& in) override;

private:
  vec3_t shape_;
  vec3_t oldShape_;
};

}

#endif //LIBOW_RESHAPE_LAYER_H
