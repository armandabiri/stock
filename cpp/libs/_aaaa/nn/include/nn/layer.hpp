#pragma once

#include <random>
#include <iostream>
#include <fstream>

#include "nn/activation_funcs.hpp"
#include "matrix/matrix.hpp"
#include "matrix/vector.hpp"
namespace nn {
class Layer {
 public:
  struct Option {
    nn::ActivationFunction::Type activationFunctionType = nn::ActivationFunction::Type::SOFTMAX;
    double dropoutRate = 1e-8;
    double learningRate = 10;
    double gradientClipThreshold = 1e6;
    double adaGradEps = 1e-6;
  };
  Layer(const size_t& inputSize, const size_t& outputSize, const Option& option)
      : inputSize_(inputSize), outputSize_(outputSize), option_(option) {
    initializeParameters();
  }

  void initializeParameters() {
    // Xavier initialization for weights_
    double limit = std::sqrt(6.0 / (inputSize_ + outputSize_));
    weights_ = matrix::Matrix(outputSize_, inputSize_);
    weights_.rand();
    weights_ *= limit;

    // Initialize biases_ to zeros
    biases_ = matrix::Vector(outputSize_, 0);

    output_ = matrix::Vector(outputSize_, 0);
    delta_ = matrix::Vector(outputSize_, 0);

    // Update learning rate using AdaGrad
    squaredWeightGradients = matrix::Matrix(outputSize_, inputSize_, 0);
    squaredBiasGradients = matrix::Vector(outputSize_, 0);
  }

  matrix::Vector forward(const matrix::Vector& input) {
    output_ = (weights_ * input) + biases_;
    output_ = nn::ActivationFunction::Apply(output_, option_.activationFunctionType);

    if (option_.dropoutRate > 0.0) {
      dropout(output_);
    }
    return output_;
  }

  void backward(const matrix::Vector& input, const matrix::Vector& outputError) {
    matrix::Vector activation_diff =
        nn::ActivationFunction::Diff(input, option_.activationFunctionType, output_);

    delta_ = activation_diff.elementwiseProd(outputError);

    // Compute weight gradients and bias gradients
    matrix::Matrix biasGradients = delta_;
    matrix::Matrix weightGradients = delta_ * input.transpose();

    // Gradient clipping
    clipGradient(weightGradients, biasGradients);

    squaredWeightGradients += weightGradients.square();
    squaredBiasGradients += biasGradients.square();

    if (option_.adaGradEps > 0.0) {
      matrix::Matrix weightLearningRate =
          option_.learningRate / (squaredWeightGradients.sqrt() + option_.adaGradEps);
      matrix::Vector biasLearningRate =
          option_.learningRate / (squaredBiasGradients.sqrt() + option_.adaGradEps);

      // Update weights and biases
      weights_ -= weightLearningRate.elementwiseProd(weightGradients);
      biases_ -= biasLearningRate.elementwiseProd(delta_);
    } else {
      weights_ -= option_.learningRate * weightGradients;
      biases_ -= option_.learningRate * biasGradients;
    }
  }

  const size_t& size() const { return inputSize_; }

  std::vector<size_t> sizes() const {
    std::vector<size_t> _sizes = {inputSize_, outputSize_};
    return _sizes;
  }
  const matrix::Matrix& getDelta() const { return delta_; }
  const matrix::Matrix& getWeights() const { return weights_; }
  const matrix::Vector& getBiases() const { return biases_; }
  const matrix::Vector& getOutput() const { return output_; }
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
  void clipGradient(matrix::Matrix weightGradients, matrix::Matrix biasGradients) {
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
  void dropout(matrix::Vector& output_) {
    for (size_t i = 0; i < output_.length(); ++i) {
      if ((double)rand() / RAND_MAX < option_.dropoutRate) {
        output_[i] = 0.0;
      }
    }
  }

  // Helper functions for saving and loading Eigen objects
  void saveMatrix(const matrix::Matrix& matrix, std::ofstream& file) const {
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
  }

  void loadMatrix(matrix::Matrix& matrix, std::ifstream& file) {
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    matrix.resize(rows, cols);
    file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(double));
  }

  void saveVector(const matrix::Vector& vector, std::ofstream& file) const {
    size_t size = vector.length();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(double));
  }

  void loadVector(matrix::Vector& vector, std::ifstream& file) {
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    vector.resize(size);
    file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(double));
  }

  size_t inputSize_;
  size_t outputSize_;

  matrix::Matrix weights_;
  matrix::Vector biases_;
  matrix::Vector output_;
  matrix::Matrix delta_;

  // Update learning rate using AdaGrad
  matrix::Matrix squaredWeightGradients;
  matrix::Vector squaredBiasGradients;

  Option option_;
};
}  // namespace nn
