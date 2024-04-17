//
// Created by zeqi.luo on 2021/1/27.
//

#ifndef ANET_DROPOUT_LAYER_H
#define ANET_DROPOUT_LAYER_H

#include "include/layer/layer.h"

namespace anet {

class DropoutLayer : public Layer {
public:
  explicit DropoutLayer(float_t p);

  void calcY(mat_t &x, mat_t &y) override;

private:
  float_t p_;
};

}

#endif //ANET_DROPOUT_LAYER_H
