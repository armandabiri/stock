#include <iostream>
#include <random>

#include "nn/cnn.hpp"

// Generate random stock data
Eigen::MatrixXd generateStockData(size_t numSamples, size_t numFeatures) {
  Eigen::MatrixXd data(numSamples, numFeatures);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (size_t i = 0; i < numSamples; ++i) {
    for (size_t j = 0; j < numFeatures; ++j) {
      data(i, j) = distribution(generator);
    }
  }

  return data;
}

// Generate random target labels
Eigen::MatrixXd generateRandomLabels(size_t numSamples, size_t numClasses) {
  Eigen::MatrixXd labels(numSamples, numClasses);
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, numClasses - 1);

  for (size_t i = 0; i < numSamples; ++i) {
    int label = distribution(generator);
    labels(i, label) = 1.0;
  }

  return labels;
}

int main() {
  // Parameters
  size_t numSamples = 1000;
  size_t numFeatures = 10;
  size_t numClasses = 3;

  // Generate fake stock data and labels
  Eigen::MatrixXd stockData = generateStockData(numSamples, numFeatures);
  Eigen::MatrixXd labels = generateRandomLabels(numSamples, numClasses);

  // Create a CNN
  nn::CNN cnn;

  // Add convolutional layers
  cnn.addConvLayer(1, 32, 3, 1, nn::ActivationFunction::Type::RELU);
  cnn.addConvLayer(32, 64, 3, 1, nn::ActivationFunction::Type::RELU);

  // Add a flatten layer
  cnn.addFlattenLayer();

  // Add fully connected layers
  cnn.addFCLayer(64 * (numFeatures - 4) * (numFeatures - 4), 128,
                 nn::ActivationFunction::Type::RELU);
  cnn.addFCLayer(128, numClasses, nn::ActivationFunction::Type::SOFTMAX);

  // Perform training
  cnn.train(stockData, labels);

  return 0;
}
