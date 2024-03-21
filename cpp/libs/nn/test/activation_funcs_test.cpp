#include "nn/activation_funcs.hpp"

#include <iostream>

int main() {
  matrix::Vector input_vector{-1, 0, 1, 2, 3};

  matrix::Vector output =
      nn::ActivationFunction::Apply(input_vector, nn::ActivationFunction::Type::RELU);
  matrix::Vector doutput =
      nn::ActivationFunction::Diff(input_vector, nn::ActivationFunction::Type::RELU);
  matrix::Vector toutput;

  toutput = output.transpose();

  std::cout << "output: " << toutput << std::endl;
  std::cout << "doutput: " << doutput.transpose() << std::endl;
  return 0;
}