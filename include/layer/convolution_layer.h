//
// Created by low on 2019-11-29.
//

#pragma once

#include "include/layer/layer.h"

#include "include/math/vec.h"

#define MakeConvLayer \
std::make_shared<ConvolutionLayer>

namespace anet {

class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer() = default;
  explicit ConvolutionLayer(vec3_t f, vec2_t s, vec2_t p);
  ~ConvolutionLayer() = default;

  std::string name() override {
    return "conv";
  }

  void initial() override;

  vec3_t outShape() override;

  void calcY(mat_t &x, mat_t &y) override;
  mat_t calcD(mat_t& x, mat_t& y, mat_t& d,
              std::vector<mat_t>& g) override;

  void writeTo(obstream& out) override;
  void readFrom(ibstream& in) override;

private:
  vec3_t f_;
  vec2_t s_;
  vec2_t p_;
  int32_t xd_;

  int32_t variable1_;
  int32_t variable2_;
};

}