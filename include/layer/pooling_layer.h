//
// Created by zeqi.luo on 2020/10/22.
//

#ifndef LIBOW_POOLING_LAYER_H
#define LIBOW_POOLING_LAYER_H

#include "include/layer/layer.h"

#include <map>
#include <thread>

namespace anet {

class PoolingLayer : public Layer {
public:
  enum Type {
    kMaxPool_Type, kAvgPool_Type
  };
  PoolingLayer() = default;
  PoolingLayer(Type type, vec2_t f, vec2_t s);

  std::string name() override {
    return "pool";
  }

  vec3_t outShape() override;
  void calcY(mat_t &x, mat_t &y) override;
  mat_t calcD(mat_t& x, mat_t& y, mat_t& d,
              std::vector<mat_t>& g) override;

  void writeTo(obstream& out) override;
  void readFrom(ibstream& in) override;

private:
  Type type_;
  vec2_t size_;
  vec2_t s_;

  Matrix<int32_t> marks;
};

}

#endif //LIBOW_POOLING_LAYER_H