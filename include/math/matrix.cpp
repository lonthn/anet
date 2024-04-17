//
// Created by zeqi.luo on 2020/12/30.
//

#include <xmmintrin.h>  // SSE
#include <cstring>

static void a_pluseq_b4(float_t* a, float_t* b, int n, int reminder) {
  __m128 a1;
  __m128 b1;
  for (int32_t i = 0; i < n; i++) {
    a1 = _mm_load_ps(a);
    b1 = _mm_load_ps(b);
    _mm_store_ps(a, _mm_add_ps(a1, b1));
    a1 = _mm_load_ps(a + 4);
    b1 = _mm_load_ps(b + 4);
    _mm_store_ps(a + 4, _mm_add_ps(a1, b1));
    a += 8;
    b += 8;
  }
  for (int32_t i = 0; i < reminder; i++)
    *(a++) += *(b++);
}


template<typename T>
Matrix<T>::Matrix(int32_t col, int32_t row, int32_t dep)
        : width_(col), height_(row), depth_(dep), area_(col * row),
          size_(col * row * dep) {
  assert(col > 0);
  assert(row > 0);
  assert(dep > 0);
  data_ = (T*) malloc(sizeof(T) * size_);
  memset((void*) data_, 0, sizeof(T) * size_);
}

template<typename T>
Matrix<T>::Matrix(int32_t col, int32_t row, int32_t dep, bool zero)
        : width_(col), height_(row), depth_(dep), area_(col * row),
          size_(col * row * dep) {
  assert(col > 0);
  assert(row > 0);
  assert(dep > 0);
  data_ = (T*) malloc(sizeof(T) * size_);
  if (zero) {
    memset((void*) data_, 0, sizeof(T) * size_);
  }
}

template<typename T>
Matrix<T>::Matrix(int32_t col, int32_t row, int32_t dep, T* data)
        : Matrix(col, row, dep) {
  assert(data);
  memcpy(data_, data, sizeof(T) * size_);
}

template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
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
}

template<typename T>
Matrix<T>::~Matrix() {
  if (data_) {
    free(data_);
    data_ = nullptr;
  }
}

template<typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
  if (this == &other) {
    return true;
  }
  if (shape3() == other.shape3()) {
    return true;
  }
  return memcmp((void*) data_, (void*) other.data_, size_ * sizeof(T)) == 0;
}

template<typename T>
Matrix <T> Matrix<T>::operator-() const {
  if (empty()) {
    return Matrix();
  }
  Matrix result(shape3());
  for (int32_t i = 0; i < size_; ++i) {
    result[i] = -data_[i];
  }
  return result;
}

template<typename T>
void Matrix<T>::operator+=(T n) {
  for (int32_t i = 0; i < size_; ++i) {
    data_[i] += n;
  }
}

template<typename T>
void Matrix<T>::operator-=(T n) {
  for (int32_t i = 0; i < size_; ++i) {
    data_[i] -= n;
  }
}

template<typename T>
void Matrix<T>::operator*=(T n) {
  for (int32_t i = 0; i < size_; ++i) {
    data_[i] *= n;
  }
}

template<typename T>
void Matrix<T>::operator/=(T n) {
  for (int32_t i = 0; i < size_; ++i) {
    data_[i] /= n;
  }
}

template<typename T>
void Matrix<T>::operator+=(const Matrix& r) {
  assert(width_ == r.width_);
  assert(height_ == r.height_);
  assert(depth_ == r.depth_);
  a_pluseq_b4(data_, r.data_, size_ / 8, size_ % 8);
}

template<typename T>
void Matrix<T>::operator-=(const Matrix& r) {
  assert(width_ == r.width_);
  assert(height_ == r.height_);
  assert(depth_ == r.depth_);

  for (int32_t i = 0; i < size_; ++i) {
    data_[i] -= r.data_[i];
  }
}

template<typename T>
void Matrix<T>::operator*=(const Matrix& r) {
  assert(width_ == r.width_);
  assert(height_ == r.height_);
  assert(depth_ == r.depth_);

  for (int32_t i = 0; i < size_; ++i) {
    data_[i] *= r.data_[i];
  }
}

template<typename T>
void Matrix<T>::operator/=(const Matrix& r) {
  assert(width_ == r.width_);
  assert(height_ == r.height_);
  assert(depth_ == r.depth_);

  for (int32_t i = 0; i < size_; ++i) {
    data_[i] /= r.data_[i];
  }
}


#define n_operator_mat(a, opt, b, c) { \
    for (int32_t i = 0; i < (b).size_; ++i) { \
        (c)[i] = (a) opt (b).get(i); \
    } \
}
#define mat_operator_n(a, opt, b, c) { \
    for (int32_t i = 0; i < (a).size_; ++i) { \
        (c)[i] = (a).get(i) opt (b); \
    } \
}
#define mat_operator_mat(a, opt, b, c) { \
    for (int32_t i = 0; i < (a).size_; ++i) { \
        (c)[i] = (a).get(i) opt (b).get(i); \
    } \
}

template<typename E>
Matrix<E> operator+(Matrix<E>&& A, E n) {
  mat_operator_n(A, +, n, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator+(const Matrix<E>& A, E n) {
  Matrix<E> res(A.shape3());
  mat_operator_n(A, +, n, res)
  return res;
}

template<typename E>
Matrix<E> operator+(E n, Matrix<E>&& A) {
  n_operator_mat(n, +, A, A)
  return A;
}

template<typename E>
Matrix<E> operator+(E n, const Matrix<E>& A) {
  Matrix<E> res(A.shape3());
  n_operator_mat(n, +, A, res)
  return res;
}

template<typename E>
Matrix<E> operator+(Matrix<E>&& A, const Matrix<E>& B) {
  mat_operator_mat(A, +, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator+(const Matrix<E>& A, Matrix<E>&& B) {
  mat_operator_mat(A, +, B, B)
  return B;
}

template<typename E>
Matrix<E> operator+(Matrix<E>&& A, Matrix<E>&& B) {
  mat_operator_mat(A, +, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator+(const Matrix<E>& A, const Matrix<E>& B) {
  Matrix<E> res(A.shape3());
  mat_operator_mat(A, +, B, res)
  return res;
}

template<typename E>
Matrix<E> operator-(Matrix<E>&& A, E n) {
  mat_operator_n(A, -, n, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator-(const Matrix<E>& A, E n) {
  Matrix<E> res(A.shape3());
  mat_operator_n(A, -, n, res)
  return res;
}

template<typename E>
Matrix<E> operator-(E n, Matrix<E>&& A) {
  n_operator_mat(n, -, A, A)
  return A;
}

template<typename E>
Matrix<E> operator-(E n, const Matrix<E>& A) {
  Matrix<E> res(A.shape3());
  n_operator_mat(n, -, A, res)
  return res;
}

template<typename E>
Matrix<E> operator-(Matrix<E>&& A, const Matrix<E>& B) {
  mat_operator_mat(A, -, B, A)
  return A;
}

template<typename E>
Matrix<E> operator-(const Matrix<E>& A, Matrix<E>&& B) {
  mat_operator_mat(A, -, B, B)
  return B;
}

template<typename E>
Matrix<E> operator-(Matrix<E>&& A, Matrix<E>&& B) {
  mat_operator_mat(A, -, B, A)
  return A;
}

template<typename E>
Matrix<E> operator-(const Matrix<E>& A, const Matrix<E>& B) {
  Matrix<E> res(A.shape3());
  mat_operator_mat(A, -, B, res)
  return res;
}

template<typename E>
Matrix<E> operator*(Matrix<E>&& A, E n) {
  mat_operator_n(A, *, n, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator*(const Matrix<E>& A, E n) {
  Matrix<E> res(A.shape3());
  mat_operator_n(A, *, n, res)
  return res;
}

template<typename E>
Matrix<E> operator*(E n, Matrix<E>&& A) {
  n_operator_mat(n, *, A, A)
  return A;
}

template<typename E>
Matrix<E> operator*(E n, const Matrix<E>& A) {
  Matrix<E> res(A.shape3());
  n_operator_mat(n, *, A, res)
  return res;
}

template<typename E>
Matrix<E> operator*(Matrix<E>&& A, const Matrix<E>& B) {
  mat_operator_mat(A, *, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator*(const Matrix<E>& A, Matrix<E>&& B) {
  mat_operator_mat(A, *, B, B)
  return std::move(B);
}

template<typename E>
Matrix<E> operator*(Matrix<E>&& A, Matrix<E>&& B) {
  mat_operator_mat(A, *, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator*(const Matrix<E>& A, const Matrix<E>& B) {
  Matrix<E> res(A.shape3());
  mat_operator_mat(A, *, B, res)
  return res;
}

template<typename E>
Matrix<E> operator/(Matrix<E>&& A, E n) {
  mat_operator_n(A, /, n, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator/(const Matrix<E>& A, E n) {
  Matrix<E> res(A.shape3());
  mat_operator_n(A, /, n, res)
  return res;
}

template<typename E>
Matrix<E> operator/(E n, Matrix<E>&& A) {
  n_operator_mat(n, /, A, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator/(E n, const Matrix<E>& A) {
  Matrix<E> res(A.shape3());
  n_operator_mat(n, /, A, res)
  return res;
}

template<typename E>
Matrix<E> operator/(Matrix<E>&& A, const Matrix<E>& B) {
  mat_operator_mat(A, /, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator/(const Matrix<E>& A, Matrix<E>&& B) {
  mat_operator_mat(A, /, B, B)
  return std::move(B);
}

template<typename E>
Matrix<E> operator/(Matrix<E>&& A, Matrix<E>&& B) {
  mat_operator_mat(A, /, B, A)
  return std::move(A);
}

template<typename E>
Matrix<E> operator/(const Matrix<E>& A, const Matrix<E>& B) {
  Matrix<E> res(A.shape3());
  mat_operator_mat(A, /, B, res)
  return res;
}

template<typename E>
Matrix<E> Pow(Matrix<E>&& A, E n) {
  assert(!A.empty());
  for (int32_t i = 0; i < A.size(); ++i) {
    A[i] = std::pow(A[i], n);
  }
  return std::move(A);
}

template<typename E>
Matrix<E> Pow(const Matrix<E>& A, E n) {
  assert(!A.empty());
  Matrix<E> temp(A.shape3());
  for (int32_t i = 0; i < A.size(); ++i) {
    temp[i] = std::pow(A.get(i), n);
  }
  return temp;
}

template<typename E>
Matrix<E> Sqrt(Matrix<E>&& A) {
  assert(!A.empty());
  for (int32_t i = 0; i < A.size(); ++i) {
    A[i] = std::sqrt(A[i]);
  }
  return std::move(A);
}

template<typename E>
Matrix<E> Sqrt(const Matrix<E>& A) {
  assert(!A.empty());
  Matrix<E> temp(A.shape3());
  for (int32_t i = 0; i < A.size(); ++i) {
    temp[i] = std::sqrt(A.get(i));
  }
  return temp;
}

template<typename E>
Matrix<E> Log(Matrix<E>&& A) {
  for (int32_t i = 0; i < A.size(); ++i) {
    A[i] = A[i] != 0 ? std::log(A[i]) : 0;
  }
  return std::move(A);
}

template<typename E>
Matrix<E> Log(const Matrix<E>& A) {
  assert(!A.empty());
  Matrix<E> temp(A.shape3());
  for (int32_t i = 0; i < A.size(); ++i) {
    temp[i] = A.get(i) != 0 ? std::log(A.get(i)) : 0;
  }
  return temp;
}

template<typename E>
Matrix<E> Exp(Matrix<E>&& A) {
  for (int32_t i = 0; i < A.size(); ++i) {
    A[i] = std::exp(A[i]);
  }
  return std::move(A);
}

template<typename E>
Matrix<E> Exp(const Matrix<E>& A) {
  assert(!A.empty());
  Matrix<E> temp(A.shape3());
  for (int32_t i = 0; i < A.size(); ++i) {
    temp[i] = std::exp(A.get(i));
  }
  return temp;
}

template<typename E>
E Sum(const Matrix<E>& A) {
  if (A.empty()) {
    return 0;
  }
  E sum = A.get(0);
  for (int32_t i = 1; i < A.size(); ++i) {
    sum += A.get(i);
  }
  return sum;
}


/**
 * matrix multiplication
 * @param other factor B
 * @return operation result
 */
template<typename T>
Matrix <T> Matrix<T>::multiply(const Matrix& other) const {
  assert(width_ == other.height_);
  assert(depth_ == other.depth_);

  Matrix temp(other.width_, height_, depth_);
  if (other.width() == 1) {
    for (int32_t z = 0; z < depth_; z++) {
      for (int32_t y = 0; y < height_; y++) {
        for (int32_t x = 0; x < width_; x++) {
          temp[z*area_ + y] += get(x, y, z) * other.data_[z*area_ + x];
        }
      }
    }
  } else {
    for (int32_t z = 0; z < depth_; ++z) {
      for (int32_t y = 0; y < height_; y++) {
        for (int32_t i = 0; i < width_; i++) {
          float_t v = get(i, y, z);
          for (int32_t x = 0; x < other.width_; x++) {
            temp(x, y, z) += v * other.get(x, i, z);
          }
        }
      }
    }
  }
  return temp;
}

template<typename T>
Matrix <T> Matrix<T>::tran() const {
  Matrix temp(height_, width_, depth_);
  for (int32_t z = 0; z < depth_; ++z) {
    for (int32_t y = 0; y < width_; ++y) {
      for (int32_t x = 0; x < height_; ++x) {
        temp(x, y, z) = get(y, x, z);
      }
    }
  }
  return temp;
}

template<typename T>
Matrix <T> Matrix<T>::rotate180() const {
  Matrix temp(shape3());
  for (int32_t z = 0; z < depth_; z++) {
    for (int32_t y = 0; y < height_; y++) {
      for (int32_t x = 0; x < width_; x++) {
        temp(width_ - 1 - x, height_ - 1 - y, z) = get(x, y, z);
      }
    }
  }
  return temp;
}

template<typename T>
Matrix <T> Matrix<T>::sum(vec3_t dir) const {
  if (empty()) {
    return Matrix();
  }
  if (dir.z == 1 && dir.y == 0 && dir.x == 0) {
    Matrix temp(depth_);
    for (int32_t z = 0; z < depth_; ++z) {
      temp[z] = 0;
      for (int32_t i = 0; i < area_; ++i) {
        temp[z] += data_[z * area_ + i];
      }
    }
    return temp;
  }
  return Matrix();
}

template<typename T>
Matrix <T> Matrix<T>::clone() const {
  return Matrix(width_, height_, depth_, data_);
}

template<typename T>
bool Matrix<T>::empty() const {
  return data_ == nullptr;
}

template<typename T>
T Matrix<T>::min() const {
  if (empty()) {
    return 0;
  }

  T min = data_[0];
  for (int32_t i = 1; i < size_; ++i) {
    if (data_[i] < min) {
      min = data_[i];
    }
  }
  return min;
}

template<typename T>
T Matrix<T>::max() const {
  assert(!empty());
  T max = data_[0];
  for (int32_t i = 1; i < size_; ++i) {
    if (data_[i] > max) {
      max = data_[i];
    }
  }
  return max;
}

template<typename T>
std::pair <T, T> Matrix<T>::minmax() const {
  assert(!empty());
  std::pair <T, T> pair = std::make_pair(data_[0], data_[0]);
  for (int32_t i = 1; i < size_; ++i) {
    if (data_[i] < pair.first) {
      pair.first = data_[i];
    } else if (data_[i] > pair.second) {
      pair.second = data_[i];
    }
  }
  return pair;
}

template<typename T>
int32_t Matrix<T>::indexOfMax() const {
  assert(!empty());
  int32_t index = 0;
  T max = data_[0];
  for (int32_t i = 1; i < size_; ++i) {
    if (data_[i] > max) {
      index = i;
      max = data_[i];
    }
  }
  return index;
}

template<typename T>
void Matrix<T>::print() const {
  for (int32_t z = 0; z < depth_; ++z) {
    for (int32_t y = 0; y < height_; ++y) {
      for (int32_t x = 0; x < width_; ++x) {
        T ele = get(x, y, z);
        std::cout << ele << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template<typename T>
void Matrix<T>::copy(Matrix& other) {
  assert(shape3() == other.shape3());
  memcpy(data_, other.data_, size_ * sizeof(float_t));
}