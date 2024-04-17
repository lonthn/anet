//
// Created by low on 2019-11-29.
//

#pragma once

#include "include/math/matrix.h"

#include <fstream>

static int8_t read_int8(std::ifstream &file) {
  int8_t n;
  file.read((char *) &n, sizeof(int8_t));
  return n;
}

static int32_t read_int32(std::ifstream &file) {
  uint32_t n;
  file.read((char *) &n, sizeof(uint32_t));
  return int32_t(
           ((n & 0xff000000) >> 24)
         | ((n & 0x00ff0000) >> 8)
         | ((n & 0x0000ff00) << 8)
         | ((n & 0x000000ff) << 24));
}

static bool read_mnist_labels(const char *file_name,
                              std::vector<mat_t> &labels) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    return false;
  }

  int32_t magic_number     = read_int32(file);
  int32_t number_of_labels = read_int32(file);

  labels.reserve(number_of_labels);

  for (int32_t i = 0; i < number_of_labels; i++) {
    labels.emplace_back(1, 10);
    mat_t &label = labels.back();
    uint8_t val = read_int8(file);
    label[val] = 1;
  }

  file.close();
  return true;
}

static bool read_mnist_images(const char *file_name,
                              float_t scale_min,
                              float_t scale_max,
                              std::vector<mat_t> &images) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    return false;
  }

  int magic_number     = read_int32(file);
  int number_of_images = read_int32(file);
  int row              = read_int32(file);
  int col              = read_int32(file);

  images.reserve(number_of_images);

  float_t scale = (scale_max - scale_min) / 255.0f;
  for (int i = 0; i < number_of_images; ++i) {
    images.emplace_back(col, row);
    mat_t &image = images.back();
    for (int j = 0; j < image.size(); ++j) {
      uint8_t val = read_int8(file);
      image[j] = float_t(val) * scale + scale_min;
    }
  }

  file.close();
  return true;
}