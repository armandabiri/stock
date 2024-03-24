
#pragma once

#include "matrix/matrix.hpp"

namespace nn {
struct AdamOption {
  bool active = false;

  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-6;
  double learning_rate = 1e-1;

  size_t nIter = 0;

  matrix::Matrix m;
  matrix::Matrix v;

  void update(matrix::Matrix& weights, const matrix::Matrix& gradients) {
    double nIterd = static_cast<double>(++nIter);
    if (m.rows() == 0) {
      m = matrix::Matrix(weights.rows(), weights.cols());
      v = matrix::Matrix(weights.rows(), weights.cols());
      return;
    }
    // Update biased first moment estimate
    m = (beta1 * m + (1 - beta1) * gradients) / (1 - std::pow(beta1, nIterd));

    // Update biased second moment estimate
    v = (beta2 * v + (1 - beta2) * gradients.pow2()) / (1 - std::pow(beta2, nIterd));

    // Update weights
    weights += learning_rate * m.elementwiseDiv(v.sqrt() + 1);
  }

  void reset() {
    nIter = 0;
    m = matrix::Matrix();
    v = matrix::Matrix();
  }

  void save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&active), sizeof(active));
    file.write(reinterpret_cast<const char*>(&beta1), sizeof(beta1));
    file.write(reinterpret_cast<const char*>(&beta2), sizeof(beta2));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    file.write(reinterpret_cast<const char*>(&nIter), sizeof(nIter));
    m.save(file);
    v.save(file);
  }

  void load(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&active), sizeof(active));
    file.read(reinterpret_cast<char*>(&beta1), sizeof(beta1));
    file.read(reinterpret_cast<char*>(&beta2), sizeof(beta2));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    file.read(reinterpret_cast<char*>(&nIter), sizeof(nIter));
    m.load(file);
    v.load(file);
  }

  void print() const {
    std::cout << "AdamOption: active=" << active << ", beta1=" << beta1 << ", beta2=" << beta2
              << ", epsilon=" << epsilon << ", learning_rate=" << learning_rate
              << ", nIter=" << nIter << std::endl;
  }
};
}  // namespace nn