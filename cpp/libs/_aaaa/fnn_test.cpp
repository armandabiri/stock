#include "nn/fnn.hpp"

#include <iostream>

int main() {
  // Create a forward neural network
  nn::FNN nn;
  // nn.loadModel("model.bin");

  bool save = true;

  nn::Layer::Option option;
  option.activationFunctionType = nn::ActivationFunction::Type::SIGMOID;
  option.learningRate = .01;
  option.dropoutRate = 0.0;
  option.gradientClipThreshold = 1e6;
  option.adaGradEps = 1e-12;

  // Define inputs and targets for training
  Eigen::MatrixXd inputs(1, 8);
  inputs << 0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.3, 0.4;
  Eigen::MatrixXd targets(1, 4);
  targets << .99, .5, .4, .6;

  nn.addLayer(8, 4, std::vector<size_t>{10, 10, 10}, option);

  // Train the network using backpropagation
  int epochs = 10000;
  nn.train(inputs, targets, epochs);

  // Forward pass through the trained network
  for (int i = 0; i < inputs.rows(); ++i) {
    Eigen::VectorXd output = nn.forward(inputs.row(i));
    std::cout << "Output: " << output.transpose() << "==>" << "Targets: " << targets.row(i)
              << std::endl;
  }

  if (save) {
    nn.saveModel("model.bin");
  }

  return 0;
}