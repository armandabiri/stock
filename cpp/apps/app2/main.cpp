#include "lstm/lstm.hpp"
#include <iostream>

int main() {
  // Define LSTM parameters
  int input_size = 3;
  int hidden_size = 4;
  double learning_rate = 0.01;
  double gradient_clip = 5.0;  // Set a threshold for gradient clipping

  // Create LSTM instance
  lstm::LSTM lstm(input_size, hidden_size, learning_rate, gradient_clip);

  // Example input sequence
  std::vector<std::vector<double>> inputs = {{0.1, 0.2, 0.3}, {0.2, 0.3, 0.4}, {0.3, 0.4, 0.5}};

  // Forward pass through LSTM
  for (const auto& input : inputs) {
    std::vector<double> hidden_state = lstm.forward(input);

    std::cout << "Hidden state: ";
    for (const auto& val : hidden_state) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<double> cell_state = lstm.get_cell_state();
    std::cout << "Cell state: ";
    for (const auto& val : cell_state) {
      std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;
  }

  return 0;
}