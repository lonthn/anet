//
// Created by zeqi.luo on 2020/10/23.
//

#ifndef LIBOW_FULL_CONNECTION_LAYER_H
#define LIBOW_FULL_CONNECTION_LAYER_H

#include "include/layer/layer.h"

namespace anet {

class FullConnectionLayer : public Layer {
public:
  FullConnectionLayer()
    : variable1_(-1)
    , variable2_(-1) { }
  explicit FullConnectionLayer(vec2_t f);

  std::string name() override {
    return "full conn";
  }

  void initial() override;
  vec3_t outShape() override;
  void calcY(mat_t& x, mat_t& y) override;
  mat_t calcD(mat_t& x, mat_t& y, mat_t& d,
              std::vector<mat_t>& g) override;

  void writeTo(obstream& out) override;
  void readFrom(ibstream& in) override;

private:
  vec2_t fShape_;
  int32_t variable1_;
  int32_t variable2_;
};

}

#endif //LIBOW_FULL_CONNECTION_LAYER_H