//
// Created by zeqi.luo on 2020/10/22.
//

#ifndef LIBOW_LAYER_H
#define LIBOW_LAYER_H

#include "include/math/matrix.h"
#include "include/util/iobyte.h"
#include "include/types.h"

#include <vector>

namespace anet {

class Network;
class Optimizer;

class Layer {
public:
  std::shared_ptr<Layer> prev_;
  std::shared_ptr<Layer> next_;

public:
  void setId(size_t id) {
    layerId_ = int(id);
  }
  void setNet(Network* net) {
    net_ = net;
  }
  void setInShape(vec3_t inShape) {
    inShape_ = inShape;
  }

  void forward(mat_t& x, std::vector<mat_t>& outs) {
    calcY(x, outs[layerId_]);
    if (next_) {
      next_->forward(outs[layerId_], outs);
    }
  }

  void backward(mat_t& der,
                std::vector<mat_t>& outs,
                std::vector<mat_t>& grads) {
    if (prev_) {
      mat_t d = calcD(outs[layerId_ - 1],
                      outs[layerId_],
                      der, grads);
      prev_->backward(d, outs, grads);
    }
  }

  virtual std::string name() {
    return "Input Layer";
  }
  virtual void initial() { }
  virtual vec3_t outShape() {
    return inShape_;
  }
  virtual void calcY(mat_t &x, mat_t &y) {
    y.copy(x);
  }
  virtual mat_t calcD(mat_t& x, mat_t& y, mat_t& d,
                      std::vector<mat_t>& g) {
    return std::move(d);
  }

public:
  virtual void writeTo(obstream& out) {
    out << layerId_;
  }
  virtual void readFrom(ibstream& in) {
    in >> layerId_;
  }

protected:
  int layerId_;
  Network* net_;
  vec3_t inShape_;
};

}

#endif //LIBOW_LAYER_H