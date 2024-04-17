//
// Created by zeqi.luo on 2020/10/23.
//

#include "include/layer/full_connection_layer.h"
#include "include/network.h"

#include "include/layer/core/full_conn_core.h"

namespace anet {

/* dx = T(w) * dy */
mat_t Tw_mul_dy(mat_t& w, mat_t& dy) {
  int32_t i, j;
  int32_t wr = w.height();
  int32_t wc = w.width();
  mat_t dx(dy.width(), wc, 1, false);

  float_t* w_ptr  = w.data();
  float_t* dx_ptr = dx.data();
  float_t* dy_ptr = dy.data();

  for (i = 0; i < wc; ++i) {
    float_t tdx = 0;
    float_t* tdy_ptr = dy_ptr;
    for (j = 0; j < wr; ++j)
      tdx += w_ptr[j * wc + i] * *(tdy_ptr++);
    *(dx_ptr++) = tdx;
  }
  return dx;
}

FullConnectionLayer::FullConnectionLayer(vec2_t f)
  : fShape_(f)
  , variable1_(-1)
  , variable2_(-1) {
}

void FullConnectionLayer::initial() {
  Layer::initial();
  vec3_t v1Shape = vec3_t(fShape_.x, fShape_.y, 1);
  variable1_ = net_->createVariable(v1Shape);
  variable2_ = net_->createVariable(vec3_t(1, fShape_.y, 1));
}

vec3_t FullConnectionLayer::outShape() {
  mat_t& w = net_->getVariable(variable1_);
  return vec3_t(1, w.height(), 1);
}

void FullConnectionLayer::calcY(mat_t &x, mat_t &y) {
  mat_t& w = net_->getVariable(variable1_);
  mat_t& b = net_->getVariable(variable2_);
  w_mult_x_plus_b(x, w, b, y);
}

mat_t FullConnectionLayer::calcD(mat_t& x, mat_t& y, mat_t& d,
                                 std::vector<mat_t>& g) {
  mat_t& w = net_->getVariable(variable1_);
  dy_mul_Tx(d, x, g[variable1_]);
  g[variable2_].copy(d);

  if (layerId_ == 1) {
    return mat_t();
  }
  mat_t dx = Tw_mul_dy(w, d);
  dx.reshape(inShape_);
  return dx;
}

void FullConnectionLayer::writeTo(obstream& out) {
  Layer::writeTo(out);
  out << fShape_.x << fShape_.y
      << variable1_
      << variable2_;
}

void FullConnectionLayer::readFrom(ibstream& in) {
  Layer::readFrom(in);
  in >> fShape_.x >> fShape_.y
     >> variable1_
     >> variable2_;
}

}