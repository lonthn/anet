//
// Created by low on 2019-11-20.
//

#pragma once

#include "include/math/vec.h"

#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>

#define for_i(n)                     \
for (int32_t i = 0; i < n; i++)

#define for_ij(n, m)                 \
for (int32_t i = 0; i < n; i++)      \
  for (int32_t j = 0; j < m; j++)

#define for_ijk(n, m, d)             \
for (int32_t i = 0; i < n; i++)      \
  for (int32_t j = 0; j < m; j++)    \
    for (int32_t k = 0; k < d; k++)

template<typename T>
class Matrix {
public:
  Matrix() = default;

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  explicit Matrix(int32_t dep)
  : Matrix(1, 1, dep) { }

  explicit Matrix(int32_t col, int32_t row)
  : Matrix(col, row, 1) { }

  explicit Matrix(vec2_t vec2)
  : Matrix(vec2.x, vec2.y, 1) { }

  explicit Matrix(vec3_t vec3)
  : Matrix(vec3.x, vec3.y, vec3.z) { }

  explicit Matrix(vec3_t vec3, bool zero)
  : Matrix(vec3.x, vec3.y, vec3.z, zero) { }

  explicit Matrix(int32_t col, int32_t row, int32_t dep);
  explicit Matrix(int32_t col, int32_t row, int32_t dep, bool zero);
  explicit Matrix(int32_t col, int32_t row, int32_t dep, T* data);
  Matrix(Matrix&& other) noexcept;

  ~Matrix();

  inline int32_t width() const { return width_; }
  inline int32_t height() const { return height_; }
  inline int32_t depth() const { return depth_; }
  inline int32_t area() const { return area_; }
  inline int32_t size() const { return size_; }
  inline vec2_t shape2() const { return vec2_t(width_, height_); }
  inline vec3_t shape3() const { return vec3_t(width_, height_, depth_); }

  inline T* data() { return data_; }

  vec3_t reshape(int32_t x, int32_t y, int32_t z, bool resize = false) {
    return reshape(vec3_t(x, y, z), resize);
  }

  vec3_t reshape(vec3_t newShape, bool resize = false) {
    vec3_t old = shape3();
    if (resize && old.size() != newShape.size()) {
      if (data_) {
        free(data_);
      }
      data_ = (T*) malloc(sizeof(float_t) * newShape.size());
      memset(data_, 0, sizeof(T) * newShape.size());
    } else {
      assert(old.size() == newShape.size());
    }
    width_ = newShape.x;
    height_ = newShape.y;
    depth_ = newShape.z;
    area_ = newShape.x * newShape.y;
    size_ = area_ * newShape.z;
    return old;
  }

  Matrix& operator=(Matrix&& other) noexcept {
    width_ = other.width_;
    height_ = other.height_;
    depth_ = other.depth_;
    area_ = other.area_;
    size_ = other.size_;
    if (data_) {
      free(data_);
    }
    data_ = other.data_;

    other.width_ = 0;
    other.height_ = 0;
    other.depth_ = 0;
    other.area_ = 0;
    other.size_ = 0;
    other.data_ = nullptr;
    return (*this);
  }

  inline T& operator[](int32_t x) {
    assert(x < size_);
    return data_[x];
  }

  inline T& operator()(int32_t z) {
    assert(z < depth_);
    return data_[z * area_];
  }

  inline T& operator()(int32_t x, int32_t y) {
    assert(x < width_ && y < height_);
    return data_[y * width_ + x];
  }

  inline T& operator()(int32_t x, int32_t y, int32_t z) {
    assert(x < width_ && y < height_ && z < depth_);
    return data_[z * area_ + y * width_ + x];
  }

  inline T get(int32_t i) const {
    return data_[i];
  }

  inline T get(int32_t x, int32_t y) {
    assert(x < width_);
    assert(y < height_);
    return data_[y * width_ + x];
  }

  inline T get(int32_t x, int32_t y, int32_t z) const {
    assert(x < width_);
    assert(y < height_);
    assert(z < depth_);
    return data_[z * area_ + y * width_ + x];
  }

  inline void setAll(T val) {
    for (int32_t i = 0; i < size_; ++i) {
      data_[i] = val;
    }
  }

  bool operator==(const Matrix& other) const;

  Matrix operator-() const;

  void operator+=(T n);
  void operator-=(T n);
  void operator*=(T n);
  void operator/=(T n);
  void operator+=(const Matrix& r);
  void operator-=(const Matrix& r);
  void operator*=(const Matrix& r);
  void operator/=(const Matrix& r);

  template<typename E>
  friend Matrix<E> operator+(Matrix<E>&& A, E n);
  template<typename E>
  friend Matrix<E> operator+(const Matrix<E>& A, E n);
  template<typename E>
  friend Matrix<E> operator+(E n, Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> operator+(E n, const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> operator+(Matrix<E>&& A, const Matrix<E>& B);
  template<typename E>
  friend Matrix<E> operator+(const Matrix<E>& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator+(Matrix<E>&& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator+(const Matrix<E>& A, const Matrix<E>& B);

  template<typename E>
  friend Matrix<E> operator-(Matrix<E>&& A, E n);
  template<typename E>
  friend Matrix<E> operator-(const Matrix<E>& A, E n);
  template<typename E>
  friend Matrix<E> operator-(E n, Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> operator-(E n, const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> operator-(Matrix<E>&& A, const Matrix<E>& B);
  template<typename E>
  friend Matrix<E> operator-(const Matrix<E>& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator-(Matrix<E>&& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator-(const Matrix<E>& A, const Matrix<E>& B);

  template<typename E>
  friend Matrix<E> operator*(Matrix<E>&& A, E n);
  template<typename E>
  friend Matrix<E> operator*(const Matrix<E>& A, E n);
  template<typename E>
  friend Matrix<E> operator*(E n, Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> operator*(E n, const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> operator*(Matrix<E>&& A, const Matrix<E>& B);
  template<typename E>
  friend Matrix<E> operator*(const Matrix<E>& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator*(Matrix<E>&& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator*(const Matrix<E>& A, const Matrix<E>& B);

  template<typename E>
  friend Matrix<E> operator/(Matrix<E>&& A, E n);
  template<typename E>
  friend Matrix<E> operator/(const Matrix<E>& A, E n);
  template<typename E>
  friend Matrix<E> operator/(E n, Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> operator/(E n, const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> operator/(Matrix<E>&& A, const Matrix<E>& B);
  template<typename E>
  friend Matrix<E> operator/(const Matrix<E>& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator/(Matrix<E>&& A, Matrix<E>&& B);
  template<typename E>
  friend Matrix<E> operator/(const Matrix<E>& A, const Matrix<E>& B);

  template<typename E>
  friend Matrix<E> Pow(Matrix<E>&& A, E n);
  template<typename E>
  friend Matrix<E> Pow(const Matrix<E>& A, E n);
  template<typename E>
  friend Matrix<E> Sqrt(Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> Sqrt(const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> Log(Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> Log(const Matrix<E>& A);
  template<typename E>
  friend Matrix<E> Exp(Matrix<E>&& A);
  template<typename E>
  friend Matrix<E> Exp(const Matrix<E>& A);
  template<typename E>
  friend E Sum(const Matrix<E>& A);

public:
  Matrix multiply(const Matrix& other) const;
  Matrix tran() const;
  Matrix rotate180() const;
  Matrix sum(vec3_t dir) const;

//  /**
//   * The kernels.depth mode by this.depth must be zero.
//   * @param kernels
//   * @param strike
//   * @param padding
//   * @param output
//   */
//  void conv3d(const Matrix& kernels,
//              const Vector2& strike,
//              const Vector2& padding,
//              Matrix& output);

  Matrix clone() const;

  bool empty() const;
  T min() const;
  T max() const;
  std::pair<T, T> minmax() const;
  int32_t indexOfMax() const;
  void print() const;

  void copy(Matrix& other);

protected:
  int32_t width_ = 0;
  int32_t height_ = 0;
  int32_t depth_ = 0;

  int32_t area_ = 0;
  int32_t size_ = 0;

  /* matrix elements */
  T* data_ = nullptr;
};

#include "matrix.cpp"