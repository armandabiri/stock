#pragma once

#include <random>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "nn/activation_funcs.hpp"

namespace nn {
class Layer {
 public:
  struct Option {
    nn::ActivationFunction::Type activationFunctionType = nn::ActivationFunction::Type::SOFTMAX;
    double dropoutRate = 1e-8;
    double learningRate = 10;
    double gradientClipThreshold = 1e6;
  };
  Layer(const size_t& inputSize, const size_t& outputSize, const Option& option)
      : inputSize_(inputSize), outputSize_(outputSize), option_(option) {
    initializeParameters();
  }

  void initializeParameters() {
    // Xavier initialization for weights_
    double limit = std::sqrt(6.0 / (inputSize_ + outputSize_));
    weights_ = Eigen::MatrixXd::Random(outputSize_, inputSize_) * limit;

    // Initialize biases_ to zeros
    biases_ = Eigen::VectorXd::Zero(outputSize_);

    output_ = Eigen::VectorXd::Zero(outputSize_);
    delta_ = Eigen::VectorXd::Zero(outputSize_);
  }

  Eigen::VectorXd forward(const Eigen::VectorXd& input) {
    output_ = (weights_ * input) + biases_;
    output_ = nn::ActivationFunction::Apply(output_, option_.activationFunctionType);

    if (option_.dropoutRate > 0.0) {
      dropout(output_);
    }
    return output_;
  }

  void backward(const Eigen::VectorXd& input, const Eigen::VectorXd& outputError) {
    Eigen::VectorXd activation_diff =
        nn::ActivationFunction::Diff(input, option_.activationFunctionType, output_);

    delta_ = activation_diff.cwiseProduct(outputError);

    // Compute weight gradients and bias gradients
    Eigen::MatrixXd biasGradients = delta_;
    Eigen::MatrixXd weightGradients = delta_ * input.transpose();

    // Gradient clipping
    clipGradient(weightGradients, biasGradients);

    // Update weights and biases
    weights_ -= option_.learningRate * weightGradients;
    biases_ -= option_.learningRate * delta_;
  }

  const size_t& size() const { return inputSize_; }

  std::vector<size_t> sizes() const {
    std::vector<size_t> _sizes = {inputSize_, outputSize_};
    return _sizes;
  }
  const Eigen::MatrixXd& getDelta() const { return delta_; }
  const Eigen::MatrixXd& getWeights() const { return weights_; }
  const Eigen::VectorXd& getBiases() const { return biases_; }
  const Eigen::VectorXd& getOutput() const { return output_; }
  const nn::ActivationFunction::Type& getActivationType() const {
    return option_.activationFunctionType;
  }

  void saveModel(std::ofstream& file) const {
    // Save layer sizes
    file.write(reinterpret_cast<const char*>(&inputSize_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&outputSize_), sizeof(size_t));

    // Save parameters
    saveMatrix(weights_, file);
    saveVector(biases_, file);
    saveMatrix(delta_, file);
    saveVector(output_, file);
    file.write(reinterpret_cast<const char*>(&option_), sizeof(Option));
  }

  void loadModel(std::ifstream& file) {
    // Load layer sizes
    file.read(reinterpret_cast<char*>(&inputSize_), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&outputSize_), sizeof(size_t));

    // Load parameters
    loadMatrix(weights_, file);
    loadVector(biases_, file);
    loadMatrix(delta_, file);
    loadVector(output_, file);
    file.read(reinterpret_cast<char*>(&option_), sizeof(Option));
  }

 private:
  void clipGradient(Eigen::MatrixXd weightGradients, Eigen::MatrixXd biasGradients) {
    // Gradient clipping
    if (option_.gradientClipThreshold > 0.0) {
      double weightGradientNorm = weightGradients.norm();
      double biasGradientNorm = biasGradients.norm();
      double clipFactor = 1.0;
      if (weightGradientNorm > option_.gradientClipThreshold) {
        clipFactor = option_.gradientClipThreshold / weightGradientNorm;
      }
      weightGradients *= clipFactor;
      if (biasGradientNorm > option_.gradientClipThreshold) {
        clipFactor = option_.gradientClipThreshold / biasGradientNorm;
      }
      biasGradients *= clipFactor;
    }
  }

  // Dropout function
  void dropout(Eigen::VectorXd& output_) {
    for (int i = 0; i < output_.size(); ++i) {
      if ((double)rand() / RAND_MAX < option_.dropoutRate) {
        output_[i] = 0.0;
      }
    }
  }

  // Helper functions for saving and loading Eigen objects
  void saveMatrix(const Eigen::MatrixXd& matrix, std::ofstream& file) const {
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
  }

  void loadMatrix(Eigen::MatrixXd& matrix, std::ifstream& file) {
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    matrix.resize(rows, cols);
    file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(double));
  }

  void saveVector(const Eigen::VectorXd& vector, std::ofstream& file) const {
    size_t size = vector.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(double));
  }

  void loadVector(Eigen::VectorXd& vector, std::ifstream& file) {
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    vector.resize(size);
    file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(double));
  }

  size_t inputSize_;
  size_t outputSize_;

  Eigen::MatrixXd weights_;
  Eigen::VectorXd biases_;
  Eigen::VectorXd output_;
  Eigen::MatrixXd delta_;

  Option option_;
};
}  // namespace nn
