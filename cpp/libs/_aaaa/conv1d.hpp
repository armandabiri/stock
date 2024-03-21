#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "nn/activation_funcs.hpp"

namespace nn {
Eigen::VectorXd Convolution(const Eigen::VectorXd& signal, const Eigen::VectorXd& kernel,
                            size_t stride = 1, size_t padding = 0) {
  size_t signalSize = signal.size();
  size_t kernelSize = kernel.size();
  size_t outputSize = (signalSize + 2 * padding - kernelSize) / stride + 1;

  Eigen::VectorXd output(outputSize);
  for (size_t i = 0; i < outputSize; ++i) {
    output(i) = 0;
    for (size_t j = 0; j < kernelSize; ++j) {
      size_t index = i * stride + j - padding;
      if (index >= 0 && index < signalSize) {
        output(i) += signal(index) * kernel(j);
      }
    }
  }
  return output;
}

class Conv1DLayer {
 public:
  struct Option {
    size_t inputSize;
    size_t outputSize;
    size_t numFilters{1};
    size_t filterSize{0};
    size_t stride{1};
    size_t padding{0};
    double learningRate = .1;
    ActivationFunction::Type activationFunctionType{ActivationFunction::Type::SIGMOID};
  };

  Conv1DLayer(const Option& option) : option_(option) {
    option_.filterSize =
        option_.inputSize - (option_.outputSize - 1) * option_.stride + 2 * option_.padding;
    initializeParameters();
  }

  void initializeParameters() {
    filters_ = Eigen::MatrixXd::Random(option_.numFilters, option_.filterSize);
    biases_ = Eigen::MatrixXd::Random(option_.numFilters, option_.outputSize);
    input_ = Eigen::VectorXd(option_.outputSize * option_.numFilters);
    output_ = Eigen::VectorXd(option_.outputSize * option_.numFilters);
  }

  Eigen::VectorXd forward(const Eigen::VectorXd& input) {
    for (size_t f = 0; f < option_.numFilters; ++f) {
      Eigen::VectorXd convOutput =
          Convolution(input, filters_.row(f), option_.stride, option_.padding);
      convOutput += biases_.row(f);
      input_.segment(f * option_.outputSize, option_.outputSize) = convOutput;
    }

    output_ = ActivationFunction::Apply(input_, option_.activationFunctionType);

    return output_;
  }

  void backward(const Eigen::VectorXd& input, const Eigen::VectorXd& outputError) {
    Eigen::VectorXd dActivation =
        ActivationFunction::Diff(input_, option_.activationFunctionType, output_);

    Eigen::VectorXd convOutputGradient = dActivation.cwiseProduct(outputError);

    Eigen::MatrixXd filterGradients = Eigen::MatrixXd::Zero(option_.numFilters, option_.filterSize);
    Eigen::MatrixXd biasGradients = Eigen::MatrixXd::Zero(option_.numFilters, option_.outputSize);

    for (size_t f = 0; f < option_.numFilters; ++f) {
      size_t startIdx = f * option_.outputSize;
      filterGradients.row(f) =
          Convolution(input, convOutputGradient.segment(startIdx, option_.outputSize),
                      option_.stride, option_.padding);
      biasGradients.row(f) = convOutputGradient.segment(startIdx, option_.outputSize);
    }

    filters_ -= option_.learningRate * filterGradients;
    biases_ -= option_.learningRate * biasGradients;
  }

  auto getFilters() const { return filters_; }
  auto getBiases() const { return biases_; }

 private:
  Option option_;

  Eigen::MatrixXd filters_;
  Eigen::MatrixXd biases_;
  Eigen::VectorXd output_;
  Eigen::VectorXd input_;
};

}  // namespace nn
