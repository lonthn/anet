//
// Created by zeqi.luo on 2020/10/22.
//

#include "include/layer/convolution_layer.h"
#include "include/network.h"

namespace anet {

static void conv(mat_t& x, mat_t& w, vec2_t s, vec2_t p, mat_t& y) {
  int32_t yd = y.depth(), xd = x.depth(), yh = y.height(), yw = y.width();
  int32_t wh = w.height(), ww = w.width(), xh = x.height(), xw = x.width();
  for (int32_t wi = 0; wi < yd; wi++) {
    for (int32_t xi = 0; xi < xd; xi++) {
      float_t* wptr = &(w(0, 0, wi * xd + xi));
      float_t* yptr = &(y(0, 0, wi));
      for (int32_t yj = 0, oy = -p.y; yj < yh; yj++, oy+=s.y) {
        for (int32_t yk = 0, ox = -p.x; yk < yw; yk++, ox+=s.x) {
          float_t  sum = 0;
          float_t* twptr = wptr;
          for (int32_t wj = 0; wj < wh; wj++) {
            for (int32_t wk = 0; wk < ww; wk++) {
              int32_t xk = ox + wk;
              int32_t xj = oy + wj;
              if (xj >= 0 && xk >= 0 && xj < xh && xk < xw) {
                sum += *(twptr) * x(xk, xj, xi);
              }
              twptr++;
            }
          }
          //y(yk, yj, wi) += sum;
          *(yptr++) += sum;
        }
      }
    }
  }
}

static void dconv(mat_t& x, mat_t& w, vec2_t& s, vec2_t& p, mat_t& dw) {
  //Vec2 size = (x.shape2() + p * 2 - w.shape2()) / s + 1;
  //mat_t Y(size.x, size.y, X.depth() * W.depth());
  int32_t wd = w.depth(), xd = x.depth(), yh = dw.height(), yw = dw.width();
  int32_t wh = w.height(), ww = w.width(), xh = x.height(), xw = x.width();
  float_t* dwptr = dw.data();
  for (int32_t wi = 0; wi < wd; wi++) {
    float_t* wptr = &(w(0, 0, wi));
    for (int32_t xi = 0; xi < xd; xi++) {
      for (int32_t yj = 0, oy = -p.y; yj < yh; ++yj, oy+=s.y) {
        for (int32_t yk = 0, ox = -p.x; yk < yw; ++yk, ox+=s.x) {
          float_t sum = 0;
          float_t* twptr = wptr;
          for (int32_t wj = 0; wj < wh; ++wj) {
            for (int32_t wk = 0; wk < ww; ++wk) {
              int32_t xj = ox + wk;
              int32_t xk = oy + wj;
              if (xk >= 0 && xj >= 0 && xk < xh && xj < xw) {
                sum += *twptr * x(xj, xk, xi);
              }
              twptr++;
            }
          }
          //dw(yk, yj, wi * xd + xi) = sum;
          *dwptr++ = sum;
        }
      }
    }
  }
}

ConvolutionLayer::ConvolutionLayer(vec3_t f, vec2_t s, vec2_t p)
  : f_(f)
  , s_(s)
  , p_(p)
  , variable1_(0)
  , variable2_(0)  {
}

void add_by_channel(mat_t& a, mat_t& b) {
  int32_t ad = a.depth(), aa = a.area();
  for (int32_t z = 0; z < ad; ++z) {
    float_t n = b[z];
    for (int32_t j = 0; j < aa; ++j) {
      int32_t i = z * aa + j;
      a[i] += n;
    }
  }
}

void ConvolutionLayer::initial() {
  Layer::initial();
  vec3_t v1Shape = vec3_t(f_.x, f_.y, f_.z * inShape_.z);
  variable1_ = net_->createVariable(v1Shape);
  variable2_ = net_->createVariable(vec3_t(1, 1, f_.z));
}

vec3_t ConvolutionLayer::outShape() {
  //        Xwh + 2P - Wwh
  // Ywh = ---------------- + 1
  //              S
  mat_t& w = net_->getVariable(variable1_);
  vec2_t size = (inShape_.shape2() + p_ * 2 - w.shape2()) / s_ + 1;
  return vec3_t{size.x, size.y, w.depth() / inShape_.z};
}

void ConvolutionLayer::calcY(mat_t &x, mat_t &y) {
  mat_t& w = net_->getVariable(variable1_);
  mat_t& b = net_->getVariable(variable2_);
  conv(x, w, s_, p_, y);
  add_by_channel(y, b);
}

mat_t ConvolutionLayer::calcD(mat_t& x, mat_t& y, mat_t& d,
                              std::vector<mat_t>& g) {
  mat_t w = net_->getVariable(variable1_).rotate180();
  dconv(x, d, s_, p_, g[variable1_]);
  g[variable2_] = d.sum(vec3_t(0, 0, 1));

  if (layerId_ == 1) {
    return mat_t{};
  }

  vec2_t p = ((x.shape2() - 1) * s_ + (w.shape2() - d.shape2())) / 2;
  vec2_t size = (d.shape2() + p_ * 2 - w.shape2()) / s_ + 1;
  mat_t dx(size.x, size.y, w.depth() / d.depth());
  conv(d, w, s_, p, dx);
  return dx;
}

void ConvolutionLayer::writeTo(obstream& out) {
  Layer::writeTo(out);

  out << f_.x << f_.y << f_.z
      << s_.x << s_.y
      << p_.x << p_.y
      << xd_
      << variable1_
      << variable2_;
}

void ConvolutionLayer::readFrom(ibstream& in) {
  Layer::readFrom(in);

  in >> f_.x >> f_.y >> f_.z
     >> s_.x >> s_.y
     >> p_.x >> p_.y
     >> xd_
     >> variable1_
     >> variable2_;
}

}