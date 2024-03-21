#pragma once

#include "nn/fnn.hpp"

namespace nn {

class CNN : public FNN {
 public:
  CNN() : FNN() {}

  void addConvLayer(const size_t& inputChannels, const size_t& outputChannels,
                    const size_t& kernelSize, size_t stride, const nn::Layer::Option& option) {
    this->addLayer(inputChannels * kernelSize, outputChannels, option);
    convParams_.push_back({inputChannels, outputChannels, kernelSize, stride});
  }

  void addFlattenLayer() {
    size_t inputSize = convParams_.back().outputChannels;
    size_t outputSize = inputSize * convParams_.back().kernelSize;
    this->addLayer(inputSize, outputSize, nn::Layer::Option());
  }

  void addFCLayer(const size_t& inputSize, const size_t& outputSize,
                  const nn::Layer::Option& option) {
    nn::Layer::Option option;
    option.activationFunctionType = activation;
    this->addLayer(inputSize, outputSize, option);
  }

 private:
  struct ConvParams {
    size_t inputChannels;
    size_t outputChannels;
    size_t kernelSize;
    size_t stride;
  };

  std::vector<ConvParams> convParams_;
};

}  // namespace nn
