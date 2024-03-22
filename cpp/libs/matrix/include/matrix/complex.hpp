#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace math {

template <typename T>
class ComplexX {
 protected:
  std::vector<T> data_;

 public:
  ComplexX() : data_(2, T(0)) {}
  explicit ComplexX(const T& rel) : data_{rel, T(0)} {}
  ComplexX(const T& rel, const T& img) : data_{rel, img} {}
  ComplexX(const ComplexX<T>& c) : data_(c.data_) {}

  ComplexX<T>& operator=(const ComplexX<T>& c) {
    data_ = c.data_;
    return *this;
  }

  ComplexX<T> conj() const {
    ComplexX<T> a = *this;
    a.data_[1] = -a.data_[1];
    return a;
  }

  T norm() const { return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1]); }

  /// overloaded operators
  ComplexX<T> operator+(const ComplexX<T>& other) const {
    ComplexX<T> result;
    result.data_[0] = data_[0] + other.data_[0];
    result.data_[1] = data_[1] + other.data_[1];
    return result;
  }

  ComplexX<T> operator-(const ComplexX<T>& other) const { return (this->operator+(-1 * other)); }

  ComplexX<T> operator*(const ComplexX<T>& other) const {
    ComplexX<T> result;
    result.data_[0] = data_[0] * other.data_[0] - data_[1] * other.data_[1];
    result.data_[1] = data_[0] * other.data_[1] + data_[1] * other.data_[0];
    return result;
  }

  ComplexX<T> operator/(const ComplexX<T>& other) const {
    return this->operator*(other.conj()) / other.norm();
  }

  ComplexX<T>& operator+=(const ComplexX<T>& other) {
    data_[0] += other.data_[0];
    data_[1] += other.data_[1];
    return *this;
  }

  ComplexX<T>& operator-=(const ComplexX<T>& other) {
    data_[0] -= other.data_[0];
    data_[1] -= other.data_[1];
    return *this;
  }

  ComplexX<T>& operator*=(const ComplexX<T>& other) {
    T real = data_[0] * other.data_[0] - data_[1] * other.data_[1];
    T imag = data_[0] * other.data_[1] + data_[1] * other.data_[0];
    data_ = {real, imag};
    return *this;
  }

  ComplexX<T> operator*(const T& other) const {
    return ComplexX<T>(data_[0] * other, data_[1] * other);
  }

  ComplexX<T> operator/(const T& other) const {
    return ComplexX<T>(data_[0] / other, data_[1] / other);
  }

  ComplexX<T>& operator/=(const ComplexX<T>& other) { return *this *= other.conj() / other.norm(); }
  ComplexX<T> operator-() const { return ComplexX<T>(-data_[0], -data_[1]); }
  ComplexX<T> operator+(const T& other) const { return ComplexX<T>(data_[0] + other, data_[1]); }
  ComplexX<T> operator-(const T& other) const { return ComplexX<T>(data_[0] - other, data_[1]); }

  friend ComplexX<T> operator+(const T& other, const ComplexX<T>& c) { return c + other; }
  friend ComplexX<T> operator-(const T& other, const ComplexX<T>& c) { return -c + other; }
  friend ComplexX<T> operator*(const T& other, const ComplexX<T>& c) {
    return ComplexX<T>(other * c.data_[0], other * c.data_[1]);
  }
  friend ComplexX<T> operator/(const T& other, const ComplexX<T>& c) {
    return other * c.conj() / c.norm();
  }

  friend std::ostream& operator<<(std::ostream& os, const ComplexX<T>& complex) {
    os << std::setw(8) << std::fixed << std::setprecision(2);
    os << complex.data_[0] << " + " << complex.data_[1] << "i";
    return os;
  }

  friend ComplexX<T> exp(const ComplexX<T>& c) {
    T exp_real = std::exp(c.data_[0]) * std::cos(c.data_[1]);
    T exp_imag = std::exp(c.data_[0]) * std::sin(c.data_[1]);
    return ComplexX<T>(exp_real, exp_imag);
  }

  friend ComplexX<T> log(const ComplexX<T>& c) {
    T log_real = std::log(c.norm());
    T log_imag = std::atan2(c.data_[1], c.data_[0]);
    return ComplexX<T>(log_real, log_imag);
  }

  friend ComplexX<T> pow(const ComplexX<T>& c, const T& p) {
    T log_real = std::log(c.norm());
    T log_imag = std::atan2(c.data_[1], c.data_[0]);
    T pow_real = std::exp(p * log_real) * std::cos(p * log_imag);
    T pow_imag = std::exp(p * log_real) * std::sin(p * log_imag);
    return ComplexX<T>(pow_real, pow_imag);
  }

  friend ComplexX<T> pow(const ComplexX<T>& c, const ComplexX<T>& p) {
    return std::exp(p * std::log(c));
  }

  friend ComplexX<T> sqrt(const ComplexX<T>& c) { return std::pow(c, 0.5); }

  friend ComplexX<T> sin(const ComplexX<T>& c) {
    T sin_real = std::sin(c.data_[0]) * std::cosh(c.data_[1]);
    T sin_imag = std::cos(c.data_[0]) * std::sinh(c.data_[1]);
    return ComplexX<T>(sin_real, sin_imag);
  }

  friend ComplexX<T> cos(const ComplexX<T>& c) {
    T cos_real = std::cos(c.data_[0]) * std::cosh(c.data_[1]);
    T cos_imag = -std::sin(c.data_[0]) * std::sinh(c.data_[1]);
    return ComplexX<T>(cos_real, cos_imag);
  }

  friend ComplexX<T> tan(const ComplexX<T>& c) { return std::sin(c) / std::cos(c); }

  friend ComplexX<T> sinh(const ComplexX<T>& c) {
    T sinh_real = std::sinh(c.data_[0]) * std::cos(c.data_[1]);
    T sinh_imag = std::cosh(c.data_[0]) * std::sin(c.data_[1]);
    return ComplexX<T>(sinh_real, sinh_imag);
  }

  friend ComplexX<T> cosh(const ComplexX<T>& c) {
    T cosh_real = std::cosh(c.data_[0]) * std::cos(c.data_[1]);
    T cosh_imag = std::sinh(c.data_[0]) * std::sin(c.data_[1]);
    return ComplexX<T>(cosh_real, cosh_imag);
  }

  friend ComplexX<T> tanh(const ComplexX<T>& c) { return std::sinh(c) / std::cosh(c); }

  friend ComplexX<T> asin(const ComplexX<T>& c) {
    return -1i * std::log(1i * c + std::sqrt(1 - c * c));
  }

  friend ComplexX<T> acos(const ComplexX<T>& c) {
    return -1i * std::log(c + 1i * std::sqrt(1 - c * c));
  }

  friend ComplexX<T> atan(const ComplexX<T>& c) { return 0.5i * std::log((1i + c) / (1i - c)); }

  friend ComplexX<T> asinh(const ComplexX<T>& c) { return std::log(c + sqrt(c * c + 1)); }

  friend ComplexX<T> acosh(const ComplexX<T>& c) { return std::log(c + sqrt(c * c - 1)); }

  friend ComplexX<T> atanh(const ComplexX<T>& c) { return 0.5 * std::log((1 + c) / (1 - c)); }

  friend ComplexX<T> conj(const ComplexX<T>& c) { return c.conj(); }

  friend T real(const ComplexX<T>& c) { return c.data_[0]; }
  friend T imag(const ComplexX<T>& c) { return c.data_[1]; }

  friend T abs(const ComplexX<T>& c) { return c.norm(); }
  friend T arg(const ComplexX<T>& c) { return atan2(c.data_[1], c.data_[0]); }

  friend ComplexX<T> polar(const T& rho, const T& theta) {
    return ComplexX<T>(rho * cos(theta), rho * sin(theta));
  }
};

typedef ComplexX<double> Complex;
typedef ComplexX<float> Complexf;

#define i Complex(0, 1)
}  // namespace math
