#pragma once

#include <vector>
#include <iostream>
#include <cstdio>

#include "nn/layer.hpp"

namespace nn {

class FNN {
 public:
  FNN() {}

  void addLayer(const nn::Layer& layer) { layers_.push_back(layer); }
  void addLayer(const size_t& input_size, const size_t& output_size,
                const std::vector<size_t>& hidden_layers,
                const std::vector<nn::Layer::Option>& options) {
    for (auto i = 0; i < hidden_layers.size(); i++) {
      if (i == 0) {
        layers_.push_back(nn::Layer(input_size, hidden_layers[i], options[i]));
      } else {
        layers_.push_back(nn::Layer(hidden_layers[i - 1], hidden_layers[i], options[i]));
      }
    }
    layers_.push_back(nn::Layer(hidden_layers.back(), output_size, options.back()));
  }

  void addLayer(const size_t& input_size, const size_t& output_size,
                const std::vector<size_t>& hidden_layers,
                const nn::Layer::Option& option = nn::Layer::Option()) {
    for (auto i = 0; i < hidden_layers.size(); i++) {
      if (i == 0) {
        layers_.push_back(nn::Layer(input_size, hidden_layers[i], option));
      } else {
        layers_.push_back(nn::Layer(hidden_layers[i - 1], hidden_layers[i], option));
      }
    }
    layers_.push_back(nn::Layer(hidden_layers.back(), output_size, option));
  }

  Eigen::VectorXd forward(const Eigen::VectorXd& inputs) {
    Eigen::VectorXd layerOutput = inputs;

    for (size_t i = 0; i < layers_.size(); ++i) {
      layerOutput = layers_[i].forward(layerOutput);
    }

    return layerOutput;
  }

  void backward(const Eigen::VectorXd& inputs, const Eigen::VectorXd& targets) {
    auto N = layers_.size() - 1;

    Eigen::VectorXd outputError = layers_[N].getOutput() - targets;

    layers_[N].backward(layers_[N - 1].getOutput(), outputError);

    for (size_t i = N - 1; i >= 1; --i) {
      outputError = layers_[i + 1].getWeights().transpose() * layers_[i + 1].getDelta();
      layers_[i].backward(layers_[i - 1].getOutput(), outputError);
    }

    outputError = layers_[1].getWeights().transpose() * layers_[1].getDelta();
    layers_[0].backward(inputs, outputError);
  }

  void train(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets,
             const int& epochs = 1000, const double& feval = 1e-6, const bool report = true) {
    if (targets.rows() != inputs.rows()) {
      std::cerr << "Error: output and inputs row size mismatch" << std::endl;
      return;
    }

    if (inputs.cols() != layers_[0].size()) {
      std::cerr << "Error: inputs and layer 0 size mismatch" << std::endl;
      return;
    }

    auto lastlayer_sizes = layers_.back().sizes();
    if (targets.cols() != lastlayer_sizes[1]) {
      std::cerr << "Error: targets and layer N size mismatch" << std::endl;
      return;
    }

    Eigen::VectorXd errors(inputs.rows());
    for (int epoch = 0; epoch < epochs; ++epoch) {
      for (int i = 0; i < inputs.rows(); ++i) {
        Eigen::VectorXd output = forward(inputs.row(i));
        backward(inputs.row(i), targets.row(i));

        if (report && (epoch % 100 == 0)) {
          Eigen::VectorXd target = targets.row(i);

          auto error = (output - target);
          errors[i] = error.norm();
        }
      }

      printf("Epoch (%d/%d): error=%e\n", epoch, epochs, errors.norm());
      if (errors.norm() < feval) {
        break;
      }
    }
  }

  auto getLayers() const { return layers_; }

  void saveModel(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for writing: " << filename << std::endl;
      return;
    }

    // Save number of layers
    size_t numLayers = layers_.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(size_t));

    // Save each layer
    for (const auto& layer : layers_) {
      layer.saveModel(file);
    }

    file.close();
  }

  void loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for reading: " << filename << std::endl;
      return;
    }

    // Clear existing layers
    layers_.clear();

    // Load number of layers
    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));

    // Load each layer
    for (size_t i = 0; i < numLayers; ++i) {
      nn::Layer layer(0, 0, nn::Layer::Option());  // Create an empty layer
      layer.loadModel(file);                       // Load parameters from file
      layers_.push_back(layer);                    // Add the loaded layer to the model
    }

    file.close();
  }

 private:
  std::vector<nn::Layer> layers_;
};
}  // namespace nn
