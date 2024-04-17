//
// Created by zeqi.luo on 2021/1/18.
//


#ifndef ANET_PARALLELIZE_LOOPER_H
#define ANET_PARALLELIZE_LOOPER_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include <future>

static void parafor(int num, std::function<void(int)>&& cb) {
  int i = 0;
  int n = num / 2;

  std::vector<std::future<void>> futures;
  for (; i < 1; i++) {
    int begin = i * n;
    int end   = begin + n;
    futures.emplace_back(std::async(std::launch::async, [=](int i) {
      for (int j = begin; j < end; j++) {
        cb(j);
      }
    }, i));
  }

  for (int j = i*n; j < num; j++)
    cb(j);
  for (auto& f : futures)
    f.wait();
}

#endif //ANET_PARALLELIZE_LOOPER_H
