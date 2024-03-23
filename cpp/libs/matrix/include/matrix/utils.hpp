#pragma once

#include <cmath>
#include <vector>

namespace math {
// range
template <typename T>
std::vector<T> range(const T& start, const T& end, const int& step = 1) {
  std::vector<T> result;
  for (T value = start; value < end; value += step) {
    result.push_back(value);
  }
  return result;
}

// linspace function
template <typename T>
std::vector<T> linspace(const T& start, const T& end, const size_t& num) {
  std::vector<T> result(num);
  T step = (end - start) / (num - 1);
  for (size_t i = 0; i < num; ++i) {
    result[i] = start + i * step;
  }
  return result;
}

// logspace function
template <typename T>
std::vector<T> logspace(const T& start, const T& end, const size_t& num) {
  std::vector<T> result(num);
  T step = (end - start) / (num - 1);
  for (size_t i = 0; i < num; ++i) {
    result[i] = std::pow(10, start + i * step);
  }
  return result;
}

}  // namespace math

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
  os << "[";
  for (size_t i = 0; i < vector.size(); ++i) {
    os << std::setw(8) << std::fixed << std::setprecision(2) << vector[i] << " ";
  }
  os << "]";
  return os;
};