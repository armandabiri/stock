#pragma once

#include <Eigen/Dense>

namespace nn {
namespace ActivationFunction {
enum class Type { RELU, SIGMOID, TANH, SOFTMAX };

Eigen::VectorXd Apply(const Eigen::VectorXd& x, const Type& T = Type::SIGMOID) {
  Eigen::VectorXd y;
  switch (T) {
    case Type::RELU:
      y = x.cwiseMax(0);
      break;
    case Type::SIGMOID:
      y = 1.0 / (1.0 + (-x.array()).exp());
      break;
    case Type::TANH:
      y = x.array().tanh();
      break;
    case Type::SOFTMAX: {
      Eigen::VectorXd expX = x.array().exp();
      y = expX / expX.sum();
      break;
    }
  }
  return y;
}

Eigen::VectorXd Diff(const Eigen::VectorXd& x, const Type& T = Type::SIGMOID,
                     const Eigen::VectorXd& f = Eigen::VectorXd(0)) {
  Eigen::VectorXd y;
  switch (T) {
    case Type::RELU:
      y = (x.array() > 0).cast<double>();
      break;
    case Type::SIGMOID:
      y = f.array() * (1 - f.array());
      break;
    case Type::TANH:
      y = 1 - x.array().tanh().square();
      break;
    case Type::SOFTMAX:
      return f.array() * (1.0 - f.array());
  }
  return y;
}
}  // namespace ActivationFunction
}  // namespace nn
