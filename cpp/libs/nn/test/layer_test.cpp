#include <iostream>

#include "nn/layer.hpp"
#include "matrix/matrix.hpp"

int main() {
  // Define input size, output size, and activation function type
  matrix::Matrix matrix;
  size_t inputSize = 2;
  size_t outputSize = 3;
  nn::Layer::Option option;
  option.activationFunctionType = nn::ActivationFunction::Type::TANH;
  option.learningRate = 1e-1;
  option.adamOptionW.active = true;
  option.adamOptionB.active = false;
  // Create a layer
  nn::Layer layer(inputSize, outputSize, option);

  // Define input vector
  matrix::Matrix input(1, inputSize, {1.0, 1.0, -3.0});
  matrix::Matrix desiredOutput(1, outputSize, {0.1, -0.95, 0.3});
  std::cout << "Desired output: " << desiredOutput << std::endl;

  size_t epochs = 10000;
  for (size_t i = 0; i < epochs; ++i) {
    auto error = desiredOutput - layer.output;

    layer.train(input, error);

    if (error.norm() < 1e-6) {
      std::cout << "Converged in " << i << " epochs\n";
      break;
    }
    layer.backward(input, error);
  }

  // Display updated weights and biases
  std::cout << "Final output:" << layer.output << "\n\n";

  std::ofstream file("test_layer.bin", std::ios::binary);
  layer.save(file);
  file.close();

  std::ifstream file2("test_layer2.bin", std::ios::binary);
  nn::Layer layer2;
  layer2.load(file2);

  std::cout << "Loaded layer output:" << layer.output << "\n\n";

  return 0;
}
