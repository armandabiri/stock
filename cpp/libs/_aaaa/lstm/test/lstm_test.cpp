#include "lstm/lstm.hpp"
#include <iostream>
#include <vector>

int main() {
  // Define LSTM parameters
  int input_size = 3;
  int hidden_size = 4;
  double learning_rate = 0.01;
  double gradient_clip = 5.0;

  // Create LSTM instance
  lstm::LSTM lstm(input_size, hidden_size, learning_rate, gradient_clip);

  // Example input sequence
  std::vector<std::vector<double>> inputs = {{0.1, 0.2, 0.3}, {0.2, 0.3, 0.4}, {0.3, 0.4, 0.5}};

  // Training loop
  int num_epochs = 10;
  for (int epoch = 1; epoch <= num_epochs; ++epoch) {
    std::cout << "Epoch " << epoch << std::endl;

    // Forward pass through LSTM to train or initialize
    for (const auto& input : inputs) {
      std::vector<double> hidden_state = lstm.forward(input);
    }

    // Increment epoch after each training iteration
    lstm.increment_epoch();
  }

  // New input sequence for prediction
  std::vector<std::vector<double>> test_inputs = {{0.4, 0.5, 0.6}, {0.5, 0.6, 0.7}};

  // Predictions
  for (const auto& input : test_inputs) {
    std::vector<double> prediction = lstm.forward(input);

    std::cout << "Input: ";
    for (const auto& val : input) {
      std::cout << val << " ";
    }
    std::cout << "| Prediction: ";
    for (const auto& val : prediction) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
