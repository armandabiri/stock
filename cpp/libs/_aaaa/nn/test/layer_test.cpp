#include <iostream>

#include "nn/layer.hpp"
#include "matrix/matrix.hpp"
int main() {
  // Define input size, output size, and activation function type
  matrix::Matrix matrix;
  size_t inputSize = 2;
  size_t outputSize = 3;
  nn::Layer::Option option;
  option.activationFunctionType = nn::ActivationFunction::Type::SIGMOID;

  // Create a layer
  nn::Layer layer(inputSize, outputSize, option);

  // Define input vector
  matrix::Vector input(inputSize);
  input.rand();

  // Forward pass
  matrix::Vector output = layer.forward(input);

  // Display output

  // Define output error (for backpropagation)
  matrix::Vector outputError{0.1, -0.2, 0.3};
  std::cout << "Input:\n" << input.transpose() << std::endl;
  std::cout << "outputError:\n" << outputError.transpose() << std::endl;

  // Backward pass
  double learningRate = 1;

  size_t epochs = 100;
  for (size_t i = 0; i < epochs; ++i) {
    output = layer.forward(input);
    layer.backward(input, outputError);
  }

  // Display updated weights and biases
  std::cout << "Final output:" << layer.getOutput().transpose() << "\n\n";
  std::cout << "\nUpdated weights:\n" << layer.getWeights() << std::endl;
  std::cout << "\nUpdated biases:\n" << layer.getBiases() << std::endl;

  return 0;
}
