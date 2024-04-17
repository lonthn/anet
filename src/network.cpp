//
// Created by zeqi.luo on 2020/10/23.
//

#include "include/network.h"
#include "include/layer/convolution_layer.h"
#include "include/layer/full_connection_layer.h"
#include "include/layer/activation_layer.h"
#include "include/layer/pooling_layer.h"
#include "include/layer/reshape_layer.h"
#include "include/optimizer/optimizer.h"
#include "include/util/iobyte.h"

#include <fstream>
#include <memory>
#include <algorithm>
#include <random>

//#define SHOW_USE_TIME

static void mat_random(mat_t& m, float_t min, float_t max) {
  auto r = [&]() -> float_t {
      return (float_t) (rand() % 10) / 10 * (max - min) + min;
  };
  for (int32_t i = 0; i < m.size(); ++i) {
    m[i] = r();
  }
}

namespace anet {

Network::Network()
  : zeroGradients_(true) {
  mutex_ = std::make_shared<std::mutex>();
  loss_ = std::make_shared<CrossEntropyFunc>();

  layers_.push_back(std::make_shared<Layer>());
  layers_.front()->setId(0);
  layers_.front()->setNet(this);
  layers_.front()->initial();
  variableInitFunc_ = [](mat_t& var) {
    mat_random(var, -0.5, 0.5);
  };
}

Network& Network::input(int32_t x, int32_t y, int32_t z) {
  inShape_ = vec3_t(x, y, z);
  inputLayer()->setInShape(inShape_);
  return *this;
}

Network& Network::add(std::shared_ptr<Layer> layer) {
  layers_.back()->next_ = layer;
  layer->prev_ = layers_.back();
  layer->setId(layers_.size());
  layer->setInShape(layer->prev_->outShape());
  layer->setNet(this);
  layer->initial();
  layers_.push_back(std::move(layer));
  return *this;
}

Network& Network::done(std::shared_ptr<Optimizer> optimizer,
                       std::shared_ptr<LossFunction> loss) {
  optimizer_ = std::move(optimizer);
  if (loss) {
    loss_ = std::move(loss);
  }
  return *this;
}

int32_t Network::createVariable(vec3_t shape) {
  int32_t id = (int32_t) variables_.size();
  variables_.emplace_back(shape);
  gradients_.emplace_back(shape);
  variableInitFunc_(variables_[id]);
  return id;
}

mat_t& Network::getVariable(int32_t id) {
  return variables_[id];
}

void Network::forEachLayers(std::function<void(Layer& layer)>&& handler) {
  for (auto& layer : layers_) {
    handler(*layer);
  }
}

void Network::forEachVariables(std::function<void(mat_t&)>&& handler) {
  for (auto& variable : variables_) {
    handler(variable);
  }
}

struct ThreadData {
  std::vector<mat_t> outs;
  std::vector<mat_t> grads;

  explicit ThreadData(Network& net) {
    outs.reserve(net.layerSize());
    grads.reserve(net.varSize());
    net.forEachLayers([&](Layer& layer) {
      outs.emplace_back(layer.outShape(), false);
    });
    net.forEachVariables([&](mat_t& var) {
      grads.emplace_back(var.shape3(), false);
    });
  }

  ~ThreadData() {
    outs.clear();
    grads.clear();
  }
};

mat_t Network::run(mat_t& x) {
  thread_local ThreadData data(*this);
  inputLayer()->forward(x, data.outs);
  return data.outs.back().clone();
}

void Network::train(std::vector<mat_t>& x,
                    std::vector<mat_t>& y,
                    int32_t epoch,
                    int32_t miniBatch,
                    TrainCallback&& perBatch) {
  int64_t seed = 0;
  auto re = std::default_random_engine(seed);
  std::vector<size_t> indexes(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    indexes[i] = i;
  }

  Optimizer &optimizer = *optimizer_;
  for (int32_t i = 0; i < epoch; i++) {
    std::shuffle(indexes.begin(), indexes.end(), re);
    std::pair<float_t, float_t> s = {0, 0};
    for (int32_t j = 0; j < indexes.size(); j += miniBatch) {
      for (int32_t k = j; k < j + miniBatch; k++) {
        forwardAndBackward(x[indexes[k]], y[indexes[k]], s);
      }
      optimizer(variables_, gradients_, float_t(miniBatch));
      clearGradients();
      perBatch(s.first / float_t(miniBatch), s.second / float_t(miniBatch));
      s = {0, 0};
    }
  }
}

void Network::forwardAndBackward(mat_t& x, mat_t& y,
                                 std::pair<float_t,float_t>& state) {
#ifdef SHOW_USE_TIME
  clock_t t1 = clock();
#endif
  thread_local ThreadData data(*this);
  forward(x, data.outs);
  backward(y, data.outs, data.grads);

  mat_t& a = data.outs.back();
  float_t loss = loss_->f(y, a);
  float_t acc = (y.indexOfMax() == a.indexOfMax()) ? 1 : 0;
  if (zeroGradients_) {
    for (size_t i = 0; i < data.grads.size(); i++) {
      gradients_[i].copy(data.grads[i]);
    }
    zeroGradients_ = false;
  } else {
    std::vector<mat_t>& grads = data.grads;
    for (size_t i = 0; i < grads.size(); i++) {
      gradients_[i] += grads[i];
    }
  }
  state.first += loss;
  state.second += acc;
#ifdef SHOW_USE_TIME
  clock_t t2 = clock() - t1;
  std::cout << "use time of forward and backward:" << t2 << std::endl;
#endif
}

void Network::forward(mat_t& x, std::vector<mat_t>& outs) {
  inputLayer()->forward(x, outs);
}

void Network::backward(mat_t& y, std::vector<mat_t>& outs,
                       std::vector<mat_t>& grads) {
  mat_t& a = outs.back();
  mat_t derivative = loss_->d(y, a);
  outputLayer()->backward(derivative, outs, grads);
}

bool Network::save(const char* path) {
  return save(std::string(path));
}

bool Network::save(const std::string& path) {
  std::ofstream ofs(path, std::ios_base::binary | std::ios_base::out);
  if (!ofs) {
    return false;
  }

  obstream out;
  writeTo(out);

  ofs.write((const char*)out.bytes(), out.length());
  ofs.close();
  return true;
}

bool Network::load(const char* path) {
  return load(std::string(path));
}

bool Network::load(const std::string& path) {
  std::ifstream ifs(path, std::ios_base::binary | std::ios_base::in);
  if (!ifs) {
    return false;
  }

  std::streampos pos = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  int64_t len = ifs.tellg();
  ifs.seekg(pos);

  uint8_t* bytes = new uint8_t[len];
  ifs.read((char*) bytes, len);
  ifs.close();

  ibstream in(bytes, len);
  readFrom(in);
  return true;
}

void Network::writeTo(obstream& out) {
  for (size_t i = 1; i < layers_.size(); i++) {
    out << layers_[i]->name();
    layers_[i]->writeTo(out);
  }
  out << std::string("end");

  out << int(variables_.size());
  for (auto& var : variables_) {
    out << var.width() << var.height() << var.depth();
    out.write(var.data(), var.size() * sizeof(float_t));
  }
}

void Network::readFrom(ibstream& in) {
  while (true) {
    std::string name = in.read_string();
    std::shared_ptr<Layer> layer;
    if (name == "convl") {
      layer = std::make_shared<ConvolutionLayer>();
    } else if (name == "pool") {
      layer = std::make_shared<PoolingLayer>();
    } else if (name == "full conn") {
      layer = std::make_shared<FullConnectionLayer>();
    } else if (name == "reshape") {
      layer = std::make_shared<ReshapeLayer>();
    } else if (name == "activation") {
      layer = std::make_shared<ActivationLayer>();
    } else if (name == "end") {
      break;
    }
    layer->setNet(this);
    layer->setId(layers_.size());
    layer->readFrom(in);
    layer->prev_ = layers_.back();
    layers_.back()->next_ = layer;
    layers_.emplace_back(std::move(layer));
  }

  int size = in.read_int32();
  for (int i = 0; i < size; i++) {
    vec3_t shape;
    in >> shape.x >> shape.y >> shape.z;
    float_t *data = (float_t*) malloc(shape.size() * sizeof(float_t));
    in.read(data, shape.size() * sizeof(float_t));
    variables_.emplace_back(shape.x, shape.y, shape.z, data);
    gradients_.emplace_back(shape);
    free(data);
  }
}

void Network::clearGradients() {
  zeroGradients_ = true;
//  for (auto& g : gradients_) {
//    g.setAll(0);
//  }
}

}