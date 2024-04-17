//
// Created by zeqi.luo on 2021/1/14.
//

#ifndef ANET_NETWORK_H
#define ANET_NETWORK_H

#include "include/layer/layer.h"
#include "include/util/parallelize_looper.h"

#include "include/math/functions.h"

#include <vector>
#include <map>
#include <mutex>
#include <functional>

namespace anet {

struct Result {
  int qualified;
  int total;

  float_t accuracy() const {
    return float_t(qualified) / float_t(total);
  }

};

class Optimizer;

class Network {
public:
  typedef std::function<void(float_t, float_t)> TrainCallback;

  explicit Network();

  Network& input(int32_t x, int32_t y, int32_t z);
  Network& add(std::shared_ptr<Layer> layer);
  Network& done(std::shared_ptr<Optimizer> optimizer,
                std::shared_ptr<LossFunction> loss = {});

  int layerSize() const {
    return layers_.size();
  }
  int varSize() const {
    return variables_.size();
  }

  std::shared_ptr<Layer> inputLayer() const {
    return layers_.front();
  }
  std::shared_ptr<Layer> outputLayer() const {
    return layers_.back();
  }
  std::shared_ptr<Layer> layer(int32_t index) const {
    return layers_[index + 1];
  }
  void forEachLayers(std::function<void(Layer&)>&& handler);
  void forEachVariables(std::function<void(mat_t&)>&& handler);

  int32_t createVariable(vec3_t shape);
  mat_t& getVariable(int32_t id);


public:
  mat_t run(mat_t& x);

  void train(std::vector<mat_t>& k,
             std::vector<mat_t>& y,
             int32_t epoch,
             int32_t miniBatch,
             TrainCallback&& perBatch);
  /**
   * @param j      (in) image
   * @param y      (in) label
   * @param state (out) the
   */
  void forwardAndBackward(mat_t& j, mat_t& y, std::pair<float_t,float_t>& state);
  /**
   * @param x     (in) image
   * @param outs (out) output of each layer
   */
  void forward(mat_t& x, std::vector<mat_t>& outs);
  /**
   * @param y      (in) label
   * @param outs   (in) forward output
   * @param grads (out) single training gradient
   */
  void backward(mat_t& y, std::vector<mat_t>& outs, std::vector<mat_t>& grads);

public:
  //void writeTo(obstream& out);
  //void readFrom(ibstream& in);

  /**
   * save the anet trained param to in local file(path + "ANet_" + name_)
   * @param path target file path
   * @return is ok
   * @note
   */
  bool save(const char* path = nullptr);
  bool save(const std::string& path);

  bool load(const char* path = nullptr);
  bool load(const std::string& path);

  void writeTo(obstream& out);
  void readFrom(ibstream& in);

private:
  void clearGradients();

private:
  bool zeroGradients_;
  std::vector<mat_t> variables_;
  std::vector<mat_t> gradients_;
  std::shared_ptr<std::mutex> mutex_;

  vec3_t inShape_;

  std::function<void(mat_t&)> variableInitFunc_;
  std::vector<std::shared_ptr<Layer>> layers_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<LossFunction> loss_;
};

}

#endif //ANET_NETWORK_H
