#include <iostream>

#include "nn/conv1d.hpp"

using namespace std;
using namespace Eigen;
using namespace nn;

int main() {
  cout << "________________________________________" << endl;
  // Input parameters
  Conv1DLayer::Option option;
  option.inputSize = 5;
  option.numFilters = 1;
  option.outputSize = 5;
  option.stride = 1;
  option.padding = 0;
  option.activationFunctionType = ActivationFunction::Type::SOFTMAX;

  // Create a sample input vector
  VectorXd input = VectorXd::LinSpaced(option.inputSize, 0, 1);
  cout << "Input: " << input.transpose() << endl;

  // VectorXd kernel(3, 1);
  // kernel.setOnes();

  // Perform convolution
  // VectorXd output = Convolution(input, kernel, 3);
  //
  // cout << "Output: " << output.transpose() << endl;

  Conv1DLayer convLayer(option);

  VectorXd output = convLayer.forward(input);
  cout << "CNN Output: " << output.transpose() << endl;

  // Create a sample target vector
  VectorXd outputError = VectorXd::LinSpaced(option.outputSize * option.numFilters, 0, 1);

  cout << "outputErrort: " << outputError.transpose() << endl;
  size_t epochs = 1000;
  for (size_t i = 0; i < epochs; ++i) {
    output = convLayer.forward(input);
    cout << "Output: " << output.transpose() << endl;
    convLayer.backward(input, outputError);
  }
  cout << "Final Output: " << output.transpose() << endl;

  return 0;
}
