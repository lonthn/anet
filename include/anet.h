//
// Created by zeqi.luo on 2020/10/23.
//

#ifndef LIBOW_ANET_H
#define LIBOW_ANET_H

#include "include/network.h"

#include "include/layer/convolution_layer.h"
#include "include/layer/pooling_layer.h"
#include "include/layer/full_connection_layer.h"
#include "include/layer/activation_layer.h"
#include "include/layer/reshape_layer.h"
#include "include/layer/dropout_layer.h"

#include "include/optimizer/gd_optimizer.h"
#include "include/optimizer/adam_optimizer.h"

#include "include/math/vec.h"
#include "include/math/matrix.h"

namespace anet {

typedef std::shared_ptr<ConvolutionLayer>    convl_layer;
typedef std::shared_ptr<PoolingLayer>        pool_layer;
typedef std::shared_ptr<FullConnectionLayer> fc_layer;
typedef std::shared_ptr<ActivationLayer>     act_layer;
typedef std::shared_ptr<ReshapeLayer>        reshape_layer;
typedef std::shared_ptr<DropoutLayer>        dropout_layer;

typedef std::shared_ptr<GDOptimizer>   gd_optimizer;
typedef std::shared_ptr<AdamOptimizer> adam_optimizer;

typedef std::shared_ptr<CrossEntropyFunc> cross_entropy_func;
typedef std::shared_ptr<QuadraticCost>    quadratic_cost_func;

/**
 * 如何构建一个网络？
 * 例:
 * Network net;
 * net.add(convl((3, 3, 5), 3)
 *    .add(max_pool(2, 2)
 *    .add(relu())
 *    ...
 *    .done(adam(0.001), cross_entropy());
 */


static convl_layer   convl(int fw, int fh, int fd, int xd);
static convl_layer   convl(int fw, int fh, int fd, int pw, int ph);
static convl_layer   convl(int fw, int fh, int fd, int pw, int ph, int sw, int sh);

static pool_layer    max_pool(int fw, int fh);
static pool_layer    max_pool(int fw, int fh, int sw, int sh);
static pool_layer    avg_pool(int fw, int fh);
static pool_layer    avg_pool(int fw, int fh, int sw, int sh);

static fc_layer      full_conn(int fw, int fh);

static reshape_layer reshape(int w, int h, int d);

static act_layer     sigmoid();
static act_layer     tanh();
static act_layer     relu();
static act_layer     softmax();


/**
 * 这里提供了2种训练优化器，当然你可以
 * 定义自己的优化器(定义一个类继承自Optimizer,
 * 并实现优化函数)
 */
static gd_optimizer   gradient_descent(float_t alpha);

static adam_optimizer adam(float_t alpha);
static adam_optimizer adam(float_t alpha, float_t beta1, float_t beta2, float_t epsilon);


/**
 * 损失函数
 */
static cross_entropy_func  cross_entropy();
static quadratic_cost_func quadratic();



convl_layer convl(int fw, int fh, int fd) {
  return convl(fw, fh, fd, 0, 0, 1, 1);
}
convl_layer convl(int fw, int fh, int fd, int pw, int ph) {
  return convl(fw, fh, fd, pw, ph, 1, 1);
}
convl_layer convl(int fw, int fh, int fd, int pw, int ph, int sw, int sh) {
  return std::make_shared<ConvolutionLayer>(vec3_t(fw, fh, fd),
                                            vec2_t(sw, sh),
                                            vec2_t(pw, ph));
}

pool_layer max_pool(int fw, int fh) {
  return max_pool(fw, fh, fw, fh);
}
pool_layer max_pool(int fw, int fh, int sw, int sh) {
  return std::make_shared<PoolingLayer>(PoolingLayer::kMaxPool_Type,
                                        vec2_t(fw, fh), vec2_t(sw, sh));
}
pool_layer avg_pool(int fw, int fh) {
  return avg_pool(fw, fh, fw, fh);
}
pool_layer avg_pool(int fw, int fh, int sw, int sh) {
  return std::make_shared<PoolingLayer>(PoolingLayer::kAvgPool_Type,
                                        vec2_t(fw, fh), vec2_t(sw, sh));
}

fc_layer full_conn(int fw, int fh) {
  return std::make_shared<FullConnectionLayer>(vec2_t(fw, fh));
}

reshape_layer reshape(int w, int h, int d) {
  return std::make_shared<ReshapeLayer>(vec3_t(w, h, d));
}

dropout_layer dropout(float_t p) {
  return std::make_shared<DropoutLayer>(p);
}

act_layer sigmoid() {
  return std::make_shared<ActivationLayer>(ActivationLayer::kSigmoid);
}
act_layer tanh() {
  return std::make_shared<ActivationLayer>(ActivationLayer::kTanH);
}
act_layer relu() {
  return std::make_shared<ActivationLayer>(ActivationLayer::kReLU);
}
act_layer softmax() {
  return std::make_shared<ActivationLayer>(ActivationLayer::kSoftmax);
}


gd_optimizer gradient_descent() {
  return std::make_shared<GDOptimizer>(0.1);
}
gd_optimizer gradient_descent(float_t alpha) {
  return std::make_shared<GDOptimizer>(alpha);
}
adam_optimizer adam() {
  return anet::adam(0.01);
}
adam_optimizer adam(float_t alpha) {
  return anet::adam(alpha, 0.9, 0.999, 1e-08);
}
adam_optimizer adam(float_t alpha, float_t beta1, float_t beta2, float_t epsilon) {
  return std::make_shared<AdamOptimizer>(alpha, beta1, beta2, epsilon);
}


cross_entropy_func cross_entropy() {
  return std::make_shared<CrossEntropyFunc>();
}
quadratic_cost_func quadratic() {
  return std::make_shared<QuadraticCost>();
}

}

#endif //LIBOW_ANET_H