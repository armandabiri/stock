#include "nn/activation_funcs.hpp"

#include <iostream>

int main() {
  Eigen::VectorXd input_vector(5);
  input_vector << -1, 0, 1, 2, 3;

  Eigen::VectorXd activated =
      nn::ActivationFunction::Apply(input_vector, nn::ActivationFunction::Type::RELU);
  Eigen::VectorXd diff =
      nn::ActivationFunction::Diff(input_vector, nn::ActivationFunction::Type::RELU);

  return 0;
}