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
  VectorX(size_t size) : MatrixX<T>(size, 1) {}
  VectorX(size_t size, T value) : MatrixX<T>(size, 1, value) {}
  // initialized with <<
  VectorX(std::initializer_list<T> list) : MatrixX<T>(list.size(), 1) {
    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      this->data_[i++][0] = *it;
    }
  }

  size_t size() const { return this->rows() * this->cols(); }

  VectorX<T> cross(const VectorX<T>& other) const {
    if (this->size() != 3 || other.size() != 3) {
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
    if (this->size() != other.size()) {
      throw std::invalid_argument("Vector sizes do not match");
    }
    VectorX<T> result(this->size());
    for (size_t i = 0; i < size(); ++i) {
      result(i) = (*this)(i) + other(i);
    }
    return result;
  }

  // friend MatrixX<T> operator*(const VectorX<T>& vector, const MatrixX<T>& matrix) {
  //   if (vector.cols() != matrix.rows()) {
  //     throw std::invalid_argument(
  //         "Vector size must match matrix number of rows for multiplication.");
  //   }

  //   MatrixX<T> result =
  //       vector *
  //       matrix;  // Resulting vector will have the same number of elements as matrix columns

  //   return result;
  // }

  // Define operator* for matrix * vector
  friend VectorX<T> operator*(const MatrixX<T>& matrix, const VectorX<T>& vector) {
    if (vector.size() != matrix.cols()) {
      throw std::invalid_argument(
          "Vector size must match matrix number of columns for multiplication.");
    }

    VectorX<T> result(
        matrix.rows());  // Resulting vector will have the same number of elements as matrix rows

    // Perform matrix-vector multiplication
    for (size_t i = 0; i < matrix.rows(); ++i) {
      T sum = 0;
      for (size_t j = 0; j < matrix.cols(); ++j) {
        sum += matrix(i, j) * vector[j];  // Dot product between i-th row of matrix and vector
      }
      result[i] = sum;
    }

    return result;
  }
};

typedef VectorX<double> Vector;
typedef VectorX<float> Vectorf;
}  // namespace matrix
