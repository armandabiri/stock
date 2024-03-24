#pragma once

#include "matrix/matrix.hpp"
#include "matrix/vector.hpp"

namespace nn {
namespace ActivationFunction {
enum class Type { RELU, SIGMOID, TANH, SOFTMAX, LINEAR };

matrix::Matrix Apply(const matrix::Matrix& input, const Type& T = Type::SIGMOID) {
  matrix::Matrix y;
  switch (T) {
    case Type::RELU:
      y = input.relu();
      break;
    case Type::SIGMOID:
      y = input.sigmoid();
      break;
    case Type::TANH:
      y = input.tanh();
      break;
    case Type::SOFTMAX: {
      y = input.softmax();
      break;
    }
    case Type::LINEAR:
      y = input;
      break;
  }
  return y;
}

matrix::Matrix Diff(const matrix::Matrix& input, const matrix::Matrix& output,
                    const Type& T = Type::SIGMOID) {
  matrix::Matrix y;
  switch (T) {
    case Type::RELU:
      y = (input > 0).cast<double>();
      break;
    case Type::SIGMOID:
      y = output.elementwiseProd(1 - output);
      break;
    case Type::TANH:
      y = 1 - output.pow2();
      break;
    case Type::SOFTMAX:
      y = output.elementwiseProd(1 - output);
    case Type::LINEAR:
      y = matrix::ones(output.rows(), output.cols());
  }
  return y;
}
}  // namespace ActivationFunction
}  // namespace nn
