//
// Created by zeqi.luo on 2020/10/22.
//

#include "include/layer/activation_layer.h"

namespace anet {

ActivationLayer::ActivationLayer(Type type)
: type_(type) {
  if (type_ == kELU) {
    alpha_ = 0.5;
  }
}

void sigmoid(mat_t& x, mat_t& y) {
  int32_t i;
  int32_t xn = x.size();
  float_t* xptr = x.data();
  float_t* yptr = y.data();
  for (i = 0; i < xn; ++i)
    *(yptr++) = 1 / (1 + exp(-(*(xptr++))));
}

void d_sigmoid(mat_t& y, mat_t& dx) {
  int32_t i;
  int32_t yn = y.size();
  float_t* yptr = y.data();
  float_t* dxptr = dx.data();
  for (i = 0; i < yn; ++i) {
    float_t ty = *(yptr++);
    *(dxptr++) = ty * (1 - ty);
  }
}

void tanh(mat_t& x, mat_t& y) {
  int32_t i;
  int32_t xn = x.size();
  float_t* xptr = x.data();
  float_t* yptr = y.data();
  for (i = 0; i < xn; ++i) {
    *(yptr++) = std::tanh(*(xptr++));
  }
}

void d_tanh(mat_t& y, mat_t& dx) {
  int32_t i;
  int32_t yn = y.size();
  float_t* yptr = y.data();
  float_t* dxptr = dx.data();
  for (i = 0; i < yn; ++i) {
    float_t ty = *(yptr++);
    *(dxptr++) = 1 - ty * ty;
  }
}

void softmax(mat_t& x, mat_t& y) {
  int32_t i;
  int32_t xn = x.size();
  float_t sum = 0;
  float_t max = x.max();
  float_t* xptr = x.data();
  float_t* yptr = y.data();
  for (i = 0; i < xn; ++i) {
    float_t& ty = *(yptr++);
    ty = std::exp(*(xptr++) - max);
    sum += ty;
  }
  yptr = y.data();
  for (i = 0; i < xn; ++i) {
    *(yptr++) /= sum;
  }
}

void ActivationLayer::calcY(mat_t &x, mat_t &y) {
  if (type_ == kReLU) {
    for_i(x.size()) {
      y[i] = std::max<float_t>(x[i], 0);
    }
  } else if (type_ == kELU) {
    for_i(x.size()) {
      y[i] = x[i] < 0 ? (alpha_ * (exp(x[i]) - 1)) : x[i];
    }
  } else if (type_ == kSigmoid) {
    sigmoid(x, y);
  } else if (type_ == kTanH) {
    tanh(x, y);
  } else if (type_ == kSoftmax) {
    softmax(x, y);
  }
}

mat_t ActivationLayer::calcD(mat_t& x, mat_t& y, mat_t& d, std::vector<mat_t>& g) {
  mat_t dx(x.shape3());
  switch (type_) {
    case kReLU:
      for_i(x.size()) {
        dx[i] = x[i] > 0 ? 1 : 0;
      }
      break;
    case kELU:
      for_i(x.size()) {
        dx[i] = x[i] < 0 ? (alpha_ * (exp(x[i]) - 1) + alpha_) : x[i];
      }
      break;
    case kSigmoid:
      d_sigmoid(y, dx);
      break;
    case kTanH:
      d_tanh(y, dx);
      break;
    case kSoftmax:
      return d.clone();
  }
  return dx * d;
}

void ActivationLayer::writeTo(obstream& out) {
  out << (int) type_ << alpha_;
}

void ActivationLayer::readFrom(ibstream& in) {
  type_ = (Type) in.read_int32();
  alpha_ = in.read_float32();
}

}