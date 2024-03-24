#include "nn/fnn.hpp"

#include <iostream>

int main() {
  // Create a forward neural network
  nn::FNN::Option trainOption;
  trainOption.epochs = 10000;
  trainOption.feval = 1e-6;
  trainOption.report = 100;

  size_t numData = 2;
  size_t numInputs = 3;
  size_t numOutputs = 2;
  nn::FNN nn(numInputs, numOutputs, trainOption);

  nn::Layer::Option option;
  option.activationFunctionType = nn::ActivationFunction::Type::TANH;
  option.learningRate = 1e-1;
  option.dropoutRate = 1e-6;
  option.gradientClipThreshold = 1e1;
  option.adamOptionW.active = true;
  option.adamOptionW.learning_rate = 1e-3;

  // Define inputs and targets for training
  matrix::Matrix inputs(numData, numInputs);
  inputs << 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.3, 0.4;

  matrix::Matrix targets(numData, numOutputs);
  targets << -.2, .5, -.2, .1, .4, .6;

  nn.addLayer({10}, option);

  // Train the network using backpropagation

  nn.train(inputs, targets);

  // Forward pass through the trained network
  auto output = nn.forward(inputs);
  std::cout << "Output: \n" << output << "\nTargets: \n" << targets << std::endl;

  nn.save("fnn_model.bin");

  nn::FNN nn2;
  nn2.load("fnn_model.bin");
  std::cout << "Output: \n" << nn2.forward(inputs) << "\nTargets: \n" << targets << std::endl;

  return 0;
}