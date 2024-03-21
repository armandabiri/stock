#include "nn/activation_funcs.hpp"

#include <iostream>

int main() {
  matrix::Vector input_vector{-1, 0, 1, 2, 3};

  auto activated = nn::ActivationFunction::Apply(input_vector, nn::ActivationFunction::Type::RELU);
  auto diff = nn::ActivationFunction::Diff(input_vector, nn::ActivationFunction::Type::RELU);

  return 0;
}