#pragma once
#include <cmath>

struct vec3 {
  float x, y, z;

  vec3() : x(0), y(0), z(0) {}
  vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  vec3 &operator+=(const vec3 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  vec3 &operator-=(const vec3 &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  vec3 &operator*=(float rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
  }

  vec3 operator+(const vec3 &rhs) const { return vec3(*this) += rhs; }
  vec3 operator-(const vec3 &rhs) const { return vec3(*this) -= rhs; }
  vec3 operator*(float rhs) const { return vec3(*this) *= rhs; }
  vec3 operator/(float rhs) const { return *this * (1.0f / rhs); }

  float dot(const vec3 &rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }

  vec3 cross(const vec3 &rhs) const {
    return {y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x};
  }

  float lengthSq() const {
    return x * x + y * y + z * z;
  }

  float length() const { return sqrtf(lengthSq()); }

  vec3 normalized() const {
    float len = length();
    return len > 0 ? *this * (1.0f / len) : vec3();
  }
};


// Represents a normalized quaternion for 3D rotation
struct rotation {
  float x, y, z, w;

  rotation() : x(0), y(0), z(0), w(1) {} // Identity rotation
  rotation(float x_, float y_, float z_, float w_)
      : x(x_), y(y_), z(z_), w(w_) {}

  // Creates a rotation from axis and angle (in radians)
  static rotation fromAxisAngle(const vec3 &axis, float angle) {
    float halfAngle = angle * 0.5f;
    float s = sinf(halfAngle);
    vec3 normalizedAxis = axis.normalized();
    return {normalizedAxis.x * s, normalizedAxis.y * s, normalizedAxis.z * s,
            cosf(halfAngle)};
  }

  // Combines two rotations (this * rhs)
  rotation operator*(const rotation &rhs) const {
    return {w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z};
  }

  // Returns the inverse rotation
  rotation inverse() const {
    return {-x, -y, -z, w}; // Simplified because we maintain normalization
  }

  // Rotates a vector by this rotation
  vec3 operator*(const vec3 &v) const {
    // Optimized version of v' = q * v * q^-1 for unit quaternions
    vec3 u = {x, y, z};
    float s = w;

    vec3 result = u.cross(v) * 2.0f;
    return v + result * s + u.cross(result);
  }

  // Linear interpolation between rotations
  // Note: This is simpler than slerp but less accurate for large angles
  rotation lerp(const rotation &rhs, float t) const {
    // Prefer shortest path
    float dot = x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
    float sign = dot >= 0 ? 1.0f : -1.0f;

    rotation result = {x + (rhs.x * sign - x) * t, y + (rhs.y * sign - y) * t,
                       z + (rhs.z * sign - z) * t, w + (rhs.w * sign - w) * t};

    // Normalize to maintain unit quaternion
    float invLen = 1.0f / sqrtf(result.x * result.x + result.y * result.y +
                                result.z * result.z + result.w * result.w);

    result.x *= invLen;
    result.y *= invLen;
    result.z *= invLen;
    result.w *= invLen;

    return result;
  }

  rotation normalized() const {
    float invLen = 1.0f / sqrtf(x * x + y * y + z * z + w * w);
    return {x * invLen, y * invLen, z * invLen, w * invLen};
  }
};