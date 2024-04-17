//
// Created by zeqi.luo on 2021/1/19.
//

#include "include/anet.h"
#include "mnist_util.h"
//#include "include/util/gemm.h"

using namespace anet;

int main() {
  // https://github.com/mnielsen/neural-networks-and-deep-learning.git
  Network net;
//  net.add(convl(5, 5, 20, 1)).add(max_pool(2, 2)).add(relu())
//     .add(convl(5, 5, 40, 20)).add(max_pool(2, 2)).add(relu())
//     .add(full_conn(40*4*4, 100)).add(relu())
//     .add(full_conn(100, 100)).add(relu())
//     .add(full_conn(100, 10)).add(softmax())
//     .done(gradient_descent(0.03), cross_entropy());

  net.input(28, 28, 1)
     .add(full_conn(784, 100)).add(relu())
     .add(full_conn(100,  10)).add(softmax())
     .done(gradient_descent(0.1), cross_entropy());

  net.input(28, 28, 1)
     .add(convl(5, 5, 20)).add(max_pool(2, 2)).add(relu())
     //.add(convl(5, 5, 40)).add(max_pool(2, 2)).add(relu())
     .add(full_conn(20*12*12, 10)).add(softmax())
     .done(gradient_descent(0.2), cross_entropy());

//  net.add(convl(5, 5, 20, 1)).add(max_pool(2, 2)).add(sigmoid())
//     .add(full_conn(20*12*12, 100)).add(sigmoid())
//     .add(full_conn(100, 10)).add(softmax())
//     .done(gradient_descent(0.1), cross_entropy());

  std::vector<mat_t> images, labels, timages, tlabels;
  read_mnist_labels("train-labels-idx1-ubyte", labels);
  read_mnist_images("train-images-idx3-ubyte", -0.5, 0.5, images);
  read_mnist_labels("t10k-labels-idx1-ubyte", tlabels);
  read_mnist_images("t10k-images-idx3-ubyte", -0.5, 0.5, timages);

  printf("a%ld, b:%ld\n", images.size(), labels.size());

  net.train(images, labels, 30, 10, [&](float_t l, float_t a) {
    static int32_t i = 0;
    static double loss = 0, accuracy = 0;
    loss += l;
    accuracy += a;
    if ((++i) % 100 == 0) {
      std::cout << "loss:" << loss/100
                << " accuracy:" << accuracy/100 << std::endl;
      loss = accuracy = 0;
    }
    if (i % 6000 == 0) {
      float_t test_accuracy = 0;
      for (int32_t j = 0; j < timages.size(); j++) {
        test_accuracy += net.run(timages[j]).indexOfMax() == tlabels[j].indexOfMax() ? 1 : 0;
      }
      std::cout << "test accuracy:" << test_accuracy/10000 << std::endl;
    }
  });

  // net.save("anet_mnist_param_1_0");
  net.save("anet_mnist_param_1_1");
  return 0;
}