#pragma once

#include <vector>
#include <iostream>
#include <cstdio>

#include "nn/layer.hpp"

namespace nn {

class FNN {
 public:
  struct Option {
    size_t epochs = 1000;
    double feval = 1e-6;
    size_t report = 100;
    Option() = default;
  };

  FNN(const size_t& numInputs = 1, const size_t& numOutputs = 1, const Option& option = Option())
      : numInputs(numInputs), numOutputs(numOutputs), option(option) {}

  void addLayer(const nn::Layer& layer) {
    if (layers.size()) {
      if (layers.back().numOutputs != layer.numInputs) {
        std::cerr << "Error: Layer size mismatch" << std::endl;
        return;
      }
    } else {
      if (numInputs != layer.numInputs) {
        std::cerr << "Error: Layer size mismatch" << std::endl;
        return;
      }
    }
  }
  void addLayer(const std::vector<size_t>& hidden_layers,
                const std::vector<nn::Layer::Option>& options) {
    for (auto i = 0; i < hidden_layers.size(); i++) {
      if (i == 0) {
        layers.push_back(nn::Layer(numInputs, hidden_layers[i], options[i]));
      } else {
        layers.push_back(nn::Layer(hidden_layers[i - 1], hidden_layers[i], options[i]));
      }
    }
    layers.push_back(nn::Layer(hidden_layers.back(), numOutputs, options.back()));
  }

  void addLayer(const std::vector<size_t>& hidden_layers,
                const nn::Layer::Option& option = nn::Layer::Option()) {
    for (auto i = 0; i < hidden_layers.size(); i++) {
      if (i == 0) {
        layers.push_back(nn::Layer(numInputs, hidden_layers[i], option));
      } else {
        layers.push_back(nn::Layer(hidden_layers[i - 1], hidden_layers[i], option));
      }
    }
    layers.push_back(nn::Layer(hidden_layers.back(), numOutputs, option));
  }

  template <typename... Args>
  void emplaceLayer(Args&&... args) {
    layers.emplace_back(std::forward<Args>(args)...);
  }

  matrix::Matrix forward(const matrix::Matrix& inputs) {
    matrix::Matrix layerOutput = inputs;

    for (auto& layer : layers) {
      layerOutput = layer.forward(layerOutput);
    }

    return layerOutput;
  }

  void backward(const matrix::Matrix& inputs, const matrix::Matrix& targets) {
    matrix::Matrix local_error = targets - layers.back().output;
    layers.back().backward((layers.rbegin() + 1)->output, local_error);

    for (size_t i = layers.size() - 2; i > 0; --i) {
      local_error = layers[i + 1].outputDelta * layers[i + 1].weights.transpose();
      layers[i].backward(layers[i].input, local_error);
    }

    local_error = layers[1].outputDelta * layers[1].weights.transpose();
    layers[0].backward(inputs, local_error);
  }

  void train(const matrix::Matrix& inputs, const matrix::Matrix& targets) {
    matrix::Matrix errors(numInputs);
    for (size_t epoch = 0; epoch < option.epochs; ++epoch) {
      for (size_t i = 0; i < inputs.rows(); ++i) {
        matrix::Matrix output = forward(inputs.row(i));
        backward(inputs.row(i), targets.row(i));

        if (option.report && (epoch % 100 == 0)) {
          matrix::Matrix target = targets.row(i);
          errors(i) = (output - target).norm();
        }
      }
      if (option.report && (epoch % option.report == 0)) {
        printf("Epoch (%zu/%zu): error=%e\n", epoch, option.epochs, errors.norm());
      }
      if (errors.norm() < option.feval) {
        break;
      }
    }
  }

  void save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for writing: " << filename << std::endl;
      return;
    }

    // Save number of layers
    size_t numLayers = layers.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(size_t));

    // Save each layer
    for (const auto& layer : layers) {
      layer.save(file);
    }

    file.write(reinterpret_cast<const char*>(&numInputs), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&numOutputs), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&option), sizeof(Option));

    file.close();
  }

  void load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for reading: " << filename << std::endl;
      return;
    }

    // Clear existing layers
    layers.clear();

    // Load number of layers
    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));

    // Load each layer
    for (size_t i = 0; i < numLayers; ++i) {
      nn::Layer layer(0, 0, nn::Layer::Option());  // Create an empty layer
      layer.load(file);                            // Load parameters from file
      layers.push_back(layer);                     // Add the loaded layer to the model
    }

    file.read(reinterpret_cast<char*>(&numInputs), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&numOutputs), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&option), sizeof(Option));

    file.close();
  }

 public:
  std::vector<nn::Layer> layers;
  Option option;
  size_t numInputs;
  size_t numOutputs;
};
}  // namespace nn
