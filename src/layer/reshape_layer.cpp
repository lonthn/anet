//
// Created by zeqi.luo on 2020/10/23.
//

#include "include/layer/reshape_layer.h"

namespace anet {

ReshapeLayer::ReshapeLayer(vec3_t shape) : shape_(shape) {
}

void ReshapeLayer::calcY(mat_t &x, mat_t &y) {
  mat_t temp = x.clone();
  oldShape_ = temp.reshape(shape_);
}

mat_t ReshapeLayer::calcD(mat_t& x, mat_t& y, mat_t& d, std::vector<mat_t>& g) {
  d.reshape(oldShape_);
  return std::move(d);
}

void ReshapeLayer::writeTo(obstream& out) {
  Layer::writeTo(out);
  out << shape_.x << shape_.y << shape_.z;
}

void ReshapeLayer::readFrom(ibstream& in) {
  Layer::readFrom(in);
  in >> shape_.x >> shape_.y >> shape_.z;
}

}