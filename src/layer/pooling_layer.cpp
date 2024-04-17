//
// Created by zeqi.luo on 2020/10/22.
//

#include "include/layer/pooling_layer.h"

#include <thread>

namespace anet {

mat_t create_der(mat_t& prevDer, vec2_t& size, vec2_t& S) {
  vec3_t outVec = prevDer.shape3();
  outVec.x = (outVec.x - 1) * S.x + size.x;
  outVec.y = (outVec.y - 1) * S.y + size.y;
  return mat_t(outVec);
}

void down_sample_max(mat_t& in, vec2_t& size, vec2_t& s,
                     Matrix<int32_t>& marks, mat_t& out) {
  marks.reshape(vec3_t(2, out.size(), 1), true);
  int32_t d = out.depth();
  int32_t h = out.height();
  int32_t w = out.height();

  float_t* outptr = out.data();
  int32_t* marksptr = marks.data();
  for (int32_t i = 0; i < d; ++i) {
    for (int32_t j = 0, oj = 0; j < h; ++j, oj += s.y) {
      for (int32_t k = 0, ok = 0; k < w; ++k, ok += s.x) {
        vec2_t mark = vec2_t(ok, oj);
        float_t max = in(ok, oj, i);
        for (int32_t fj = 0; fj < size.y; ++fj) {
          for (int32_t fi = 0; fi < size.x; ++fi) {
            float_t num = in(fi + ok, fj + oj, i);
            if (num > max) {
              max = num;
              mark.x = fi + ok;
              mark.y = fj + oj;
            }
          }
        }
        *(outptr++) = max;
        *(marksptr++) = mark.x;
        *(marksptr++) = mark.y;
      }
    }
  }
}

mat_t up_sample_max(mat_t& der, vec2_t& size, vec2_t& S, Matrix<int32_t>& marks) {
  mat_t prevDer = create_der(der, size, S);
  int32_t d = der.depth(), a = der.area();
  int32_t* marksptr = marks.data();
  float_t* derptr = der.data();
  for (int32_t z = 0; z < d; z++) {
    for (int32_t i = 0; i < a; i++) {
      int32_t markX = *(marksptr++);
      int32_t markY = *(marksptr++);
      prevDer(markX, markY, z) = *(derptr++);
    }
  }
  return prevDer;
}

void down_sample_avg(mat_t& in, vec2_t& size, vec2_t& S, mat_t& out) {
  float_t num(size.x * size.y);
  for (int32_t z = 0; z < out.depth(); ++z) {
    for (int32_t y = 0, offY = 0; y < out.height(); ++y, offY += S.y) {
      for (int32_t x = 0, offX = 0; x < out.width(); ++x, offX += S.x) {
        float_t sum = 0;
        for (int32_t fj = 0; fj < size.y; fj++) {
          for (int32_t fi = 0; fi < size.x; fi++) {
            sum += in(fi + offX, fj + offY);
          }
        }
        out(x, y, z) = sum / num;
      }
    }
  }
}

mat_t up_sample_avg(mat_t& prevDer, vec2_t& size, vec2_t& S) {
  mat_t der = create_der(prevDer, size, S);

  float_t num(size.x * size.y);
  for (int32_t z = 0; z < prevDer.depth(); ++z) {
    for (int32_t y = 0, offY = 0; y < prevDer.height(); ++y, offY += S.y) {
      for (int32_t x = 0, offX = 0; x < prevDer.width(); ++x, offX += S.x) {
        float_t val = prevDer(x, y, z) / num;
        for (int32_t fj = 0; fj < size.y; fj++) {
          for (int32_t fi = 0; fi < size.x; fi++) {
            der(fi + offX, fj + offY, z) = val;
          }
        }
      }
    }
  }
  return der;
}

PoolingLayer::PoolingLayer(PoolingLayer::Type type, vec2_t f, vec2_t s/* = {-1, -1}*/)
        : type_(type), size_(f), s_(s) {
  if (s_.x == -1) {
    s_.x = size_.x;
  }
  if (s_.y == -1) {
    s_.y = size_.y;
  }
}

vec3_t PoolingLayer::outShape() {
  assert((inShape_.x - size_.x) % s_.x == 0);
  assert((inShape_.y - size_.y) % s_.y == 0);

  vec3_t outShape;
  outShape.x = (inShape_.x - size_.x) / s_.x + 1;
  outShape.y = (inShape_.y - size_.y) / s_.y + 1;
  outShape.z = inShape_.z;
  return outShape;
}

void PoolingLayer::calcY(mat_t &x, mat_t &y) {
  if (type_ == kMaxPool_Type) {
    down_sample_max(x, size_, s_, marks, y);
  } else {
    down_sample_avg(x, size_, s_, y);
  }
}

mat_t PoolingLayer::calcD(mat_t& x, mat_t& y, mat_t& d, std::vector<mat_t>& g) {
  if (type_ == kMaxPool_Type) {
    return up_sample_max(d, size_, s_, marks);
  } else {
    return up_sample_avg(d, size_, s_);
  }
}

void PoolingLayer::writeTo(obstream& out) {
  Layer::writeTo(out);
  out << (int)type_
      << size_.x << size_.y
      << s_.x << s_.y;
}

void PoolingLayer::readFrom(ibstream& in) {
  Layer::readFrom(in);
  type_ = (Type)in.read_int32();
  in >> size_.x >> size_.y
     >> s_.x >> s_.y;
}

}