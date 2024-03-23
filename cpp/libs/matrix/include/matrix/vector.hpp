#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>

#include "matrix/matrix.hpp"

namespace matrix {

template <typename T = double>
class VectorX : public MatrixX<T> {
 public:
  // Constructor
  VectorX() : MatrixX<T>() {}
  VectorX(const size_t& size1, const T& value = 0, const size_t& size2 = 1)
      : MatrixX<T>(size1, size2, value) {}

  VectorX(std::initializer_list<T> list) : MatrixX<T>(list.size(), 1) {
    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      this->data_[i++] = *it;
    }
  }

  VectorX<T> cross(const VectorX<T>& other) const {
    if (this->numel() != 3 || other.numel() != 3) {
      throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }

    VectorX<T> result;
    result = other;

    result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
    result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
    result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);
    return result;
  }

  void resize(size_t size) { this->resize(size, 1); }

  /// overloaded operators
  VectorX<T> operator+(const VectorX<T>& other) const {
    if (this->numel() != other.numel()) {
      throw std::invalid_argument("Vector sizes do not match");
    }
    VectorX<T> result(this->numel());
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = (*this)(i) + other(i);
    }
    return result;
  }

  // Define operator* for matrix * vector
  friend VectorX<T> operator*(const MatrixX<T>& matrix, const VectorX<T>& vector) {
    if (vector.rows_ != matrix.cols()) {
      throw std::invalid_argument(
          "Vector size must match matrix number of columns for multiplication.");
    }

    VectorX<T> result(
        matrix.rows());  // Resulting vector will have the same number of elements as matrix rows

    // Perform matrix-vector multiplication
    for (size_t i = 0; i < matrix.rows(); ++i) {
      T sum = 0;
      for (size_t j = 0; j < matrix.cols(); ++j) {
        sum += matrix(i, j) * vector(j);  // Dot product between i-th row of matrix and vector
      }
      result(i) = sum;
    }

    return result;
  }

  // transpose
  VectorX<T> transpose() const {
    MatrixX<T> result(this->cols_, this->rows_);
    for (size_t i = 0; i < this->numel(); ++i) {
      result(i) = (*this)(i);
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const matrix::VectorX<T>& vector) {
    os << std::setw(8) << std::fixed << std::setprecision(2);
    os << "[";
    if (vector.cols() == 1) {
      for (size_t i = 0; i < vector.rows(); ++i) {
        std::cout << vector(i) << ", ";
      }
    } else {
      for (size_t i = 0; i < vector.cols(); ++i) {
        std::cout << vector(i) << "; ";
      }
    }
    os << "]";
    return os;
  };
};

typedef VectorX<double> Vector;
typedef VectorX<float> Vectorf;
};  // namespace matrix
