#include <iostream>

#include "nn/layer.hpp"

int main() {
  // Define input size, output size, and activation function type
  size_t inputSize = 2;
  size_t outputSize = 3;
  nn::Layer::Option option;
  option.activationFunctionType = nn::ActivationFunction::Type::SIGMOID;

  // Create a layer
  nn::Layer layer(inputSize, outputSize, option);

  // Define input vector
  Eigen::VectorXd input(inputSize);
  input << 0.5, -0.3;

  // Forward pass
  Eigen::VectorXd output = layer.forward(input);

  // Display output
  std::cout << "Output after forward pass:\n" << output << std::endl;

  // Define output error (for backpropagation)
  Eigen::VectorXd outputError(outputSize);
  outputError << 0.1, -0.2, 0.3;

  // Backward pass
  double learningRate = 0.01;
  layer.backward(input, outputError);

  // Display updated weights and biases
  std::cout << "\nUpdated weights:\n" << layer.getWeights() << std::endl;
  std::cout << "\nUpdated biases:\n" << layer.getBiases() << std::endl;

  return 0;
}
