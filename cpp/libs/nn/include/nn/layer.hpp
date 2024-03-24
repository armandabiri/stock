#pragma once

#include <random>
#include <iostream>
#include <fstream>

#include "nn/activation_funcs.hpp"
#include "matrix/matrix.hpp"
#include "matrix/vector.hpp"
#include "nn/adam.hpp"

namespace nn {
class Layer {
 public:
  struct Option {
    nn::ActivationFunction::Type activationFunctionType = nn::ActivationFunction::Type::TANH;
    double dropoutRate = 1e-8;
    double learningRate = 1e-3;
    double gradientClipThreshold = 1e3;
    AdamOption adamOptionW;
    AdamOption adamOptionB;

    Option() = default;
    void save(std::ofstream& file) const {
      file.write(reinterpret_cast<const char*>(&dropoutRate), sizeof(double));
      file.write(reinterpret_cast<const char*>(&learningRate), sizeof(double));
      file.write(reinterpret_cast<const char*>(&gradientClipThreshold), sizeof(double));
      file.write(reinterpret_cast<const char*>(&activationFunctionType),
                 sizeof(nn::ActivationFunction::Type));
      adamOptionW.save(file);
      adamOptionB.save(file);
    }

    void load(std::ifstream& file) {
      file.read(reinterpret_cast<char*>(&dropoutRate), sizeof(double));
      file.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
      file.read(reinterpret_cast<char*>(&gradientClipThreshold), sizeof(double));
      file.read(reinterpret_cast<char*>(&activationFunctionType),
                sizeof(nn::ActivationFunction::Type));
      adamOptionW.load(file);
      adamOptionB.load(file);
    }
  };

  Layer() = default;
  Layer(const size_t& numInputs, const size_t& numOutputs, const Option& option)
      : numInputs(numInputs), numOutputs(numOutputs), option(option) {
    initializeParameters();
  }

  matrix::Matrix forward(const matrix::Matrix& input) {
    this->input = input;
    output = (input * weights).rowPlus(biases);
    output = nn::ActivationFunction::Apply(output, option.activationFunctionType);

    if (option.dropoutRate > 0.0) {
      output.round(option.dropoutRate);
    }
    return output;
  }

  void backward(const matrix::Matrix& input, const matrix::Matrix& outputError) {
    matrix::Matrix activation_diff =
        nn::ActivationFunction::Diff(input, output, option.activationFunctionType);

    outputDelta = outputError.elementwiseProd(activation_diff);

    // Compute weight gradients and bias gradients
    matrix::Matrix biasGradients = outputDelta;
    matrix::Matrix weightGradients = input.transpose() * outputDelta;

    // Gradient clipping
    clipGradient(weightGradients, biasGradients);

    if (option.adamOptionW.active) {
      option.adamOptionW.update(weights, weightGradients);
    } else {
      weights += option.learningRate * weightGradients;
    }

    if (option.adamOptionB.active) {
      option.adamOptionB.update(biases, biasGradients);
    } else {
      biases += option.learningRate * biasGradients;
    }
  }

  void train(const matrix::Matrix& input, const matrix::Matrix& outputError) {
    forward(input);
    backward(input, outputError);
  }

  // operator overloading Operator=
  Layer& operator=(const Layer& layer) {
    numInputs = layer.numInputs;
    numOutputs = layer.numOutputs;
    weights = layer.weights;
    biases = layer.biases;
    output = layer.output;
    outputDelta = layer.outputDelta;
    option = layer.option;
    return *this;
  }

  void save(std::ofstream& file) const {
    // Save layer sizes
    file.write(reinterpret_cast<const char*>(&numInputs), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&numOutputs), sizeof(size_t));

    weights.save(file);
    biases.save(file);
    input.save(file);
    outputDelta.save(file);
    output.save(file);
    option.save(file);
  }

  void load(std::ifstream& file) {
    // Load layer sizes
    file.read(reinterpret_cast<char*>(&numInputs), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&numOutputs), sizeof(size_t));

    // Load parameters
    weights.load(file);
    biases.load(file);
    input.load(file);
    outputDelta.load(file);
    output.load(file);
    option.load(file);
  }

  size_t numInputs;
  size_t numOutputs;
  matrix::Matrix weights;
  matrix::Matrix biases;
  matrix::Matrix output;
  matrix::Matrix input;
  matrix::Matrix outputDelta;

  Option option;

 private:
  void initializeParameters() {
    // Xavier initialization for weights
    weights = matrix::Matrix(numInputs, numOutputs);
    weights.rand();

    // Initialize biases to zeros
    biases = matrix::Matrix(1, numOutputs);
    biases.rand();

    input = matrix::Matrix(1, numInputs);
    output = matrix::Matrix(1, numOutputs);
    outputDelta = matrix::Matrix(1, numOutputs);
  }

  void clipGradient(matrix::Matrix weightGradients, matrix::Matrix biasGradients) {
    // Gradient clipping
    if (option.gradientClipThreshold > 0.0) {
      double weightGradientNorm = weightGradients.norm();
      double biasGradientNorm = biasGradients.norm();
      double clipFactor = 1.0;
      if (weightGradientNorm > option.gradientClipThreshold) {
        clipFactor = option.gradientClipThreshold / weightGradientNorm;
      }
      weightGradients *= clipFactor;
      if (biasGradientNorm > option.gradientClipThreshold) {
        clipFactor = option.gradientClipThreshold / biasGradientNorm;
      }
      biasGradients *= clipFactor;
    }
  }
};
}  // namespace nn
