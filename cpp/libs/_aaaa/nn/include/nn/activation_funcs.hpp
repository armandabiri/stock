#pragma once

#include "matrix/matrix.hpp"
#include "matrix/vector.hpp"

namespace nn {
namespace ActivationFunction {
enum class Type { RELU, SIGMOID, TANH, SOFTMAX };

matrix::Matrix Apply(const matrix::Vector& x, const Type& T = Type::SIGMOID) {
  matrix::Matrix y;
  switch (T) {
    case Type::RELU:
      y = x.relu();
      break;
    case Type::SIGMOID:
      y = x.sigmoid();
      break;
    case Type::TANH:
      y = x.tanh();
      break;
    case Type::SOFTMAX: {
      y = x.softmax();
      break;
    }
  }
  return y;
}

matrix::Matrix Diff(const matrix::Vector& x, const Type& T = Type::SIGMOID,
                    const matrix::Vector& f = matrix::Vector(0)) {
  matrix::Matrix y;
  switch (T) {
    case Type::RELU:
      y = (x > 0).cast<double>();
      break;
    case Type::SIGMOID:
      y = f * (1 - f);
      break;
    case Type::TANH:
      y = 1 - x.tanh().square();
      break;
    case Type::SOFTMAX:
      return f * (1.0 - f);
  }
  return y;
}
}  // namespace ActivationFunction
}  // namespace nn
