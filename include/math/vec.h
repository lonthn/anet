//
// Created by zeqi.luo on 2020/10/22.
//


#ifndef LIBOW_VEC_H
#define LIBOW_VEC_H

#include <cstdint>

typedef struct Vector2 {
  int32_t x = 0;
  int32_t y = 0;

  Vector2() = default;

  Vector2(int32_t pX, int32_t pY)
    : x(pX), y(pY) { }

  int32_t size() const {
    return x * y;
  }

  Vector2 operator+(const Vector2& other) const {
    return {x + other.x, y + other.y};
  }

  Vector2 operator-(const Vector2& other) const {
    return {x - other.x, y - other.y};
  }

  Vector2 operator*(const Vector2& other) const {
    return {x * other.x, y * other.y};
  }

  Vector2 operator/(const Vector2& other) const {
    return {x / other.x, y / other.y};
  }

  Vector2 operator+(int32_t num) const {
    return {x + num, y + num};
  }

  Vector2 operator-(int32_t num) const {
    return {x - num, y - num};
  }

  Vector2 operator*(int32_t num) const {
    return {x * num, y * num};
  }

  Vector2 operator/(int32_t num) const {
    return {x / num, y / num};
  }
} vec2_t;

typedef struct Vector3 {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;

  Vector3() = default;

  Vector3(int32_t pX, int32_t pY, int32_t pZ) : x(pX), y(pY), z(pZ) { }

  Vector2 shape2() {
    return {x, y};
  }

  int32_t size() const {
    return x * y * z;
  }

  bool operator==(const Vector3& vec3) const {
    return (x == vec3.x && y == vec3.y && z == vec3.z);
  }

  bool operator!=(const Vector3& vec3) const {
    return (x != vec3.x || y != vec3.y || z != vec3.z);
  }
} vec3_t;

#endif //LIBOW_VEC_H
