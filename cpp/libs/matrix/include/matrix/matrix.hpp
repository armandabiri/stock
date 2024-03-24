#pragma once

#include <tuple>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iomanip>  // for std::setw

#include "matrix/Complex.hpp"
#include "matrix/utils.hpp"

#define TRANSFORM_ENABLED 1           // Set to 1 to enable transform, 0 to disable
#define PARALLEL_ENABLED 1            // Set to 1 to enable parallel computing, 0 to disable
#define STARSSAN_ENABLED 1            // Set to 1 to enable Strassen algorithm, 0 to disable
#define DIVIDE_AND_CONQUER_ENABLED 1  // Set to 1 to enable divide and conquer, 0 to disable

#if PARALLEL_ENABLED > 0
#define PARALLEL_FOR_COLLAPSE(N) \
    _Pragma(STRINGIFY(omp parallel for collapse(N)))
#define STRINGIFY(x) #x
#else
#define PARALLEL_FOR_COLLAPSE(N)
#endif

namespace matrix {
template <typename T>
class VectorX;
template <typename T = double>
class MatrixX {
 public:
  // Constructor
  MatrixX() : rows_(0), cols_(0), length_{0} {}
  MatrixX(const size_t& rows, const size_t& cols) : rows_(rows), cols_(cols) {
    length_ = rows_ * cols_;
    data_.resize(length_);
  };
  MatrixX(const size_t& rows, const size_t& cols, T value) : MatrixX(rows, cols) {
    std::fill(data_.begin(), data_.end(), value);
  }

  MatrixX(const size_t& rows, const size_t& cols, const std::initializer_list<T>& list)
      : rows_(rows), cols_(cols), length_(rows * cols), data_(list) {}

  MatrixX(const std::initializer_list<T>& list) : MatrixX(1, list.size(), list) {}
  MatrixX(const std::initializer_list<std::initializer_list<T>>& list) {
    rows_ = list.size();
    cols_ = (rows_ > 0) ? list.begin()->size() : 0;
    length_ = rows_ * cols_;

    data_.resize(length_);

    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      if (it->size() != cols_) {
        throw std::invalid_argument("Initializer list size does not match matrix size");
      }

      // Use std::copy to copy elements of each row to the data_ vector
      std::copy(it->begin(), it->end(), data_.begin() + i);
      i += cols_;
    }
  }

  MatrixX<T>& operator<<(const std::vector<T>& values) {
    if (values.size() < length_) {
      throw std::invalid_argument("Incorrect number of values for initialization");
    }

    // Copy the values to the data_ vector
    for (size_t i = 0; i < length_; ++i) {
      data_[i] = values[i];
    }

    return *this;
  }

  MatrixX<T>& operator<<(std::initializer_list<T> values) {
    for (const auto& value : values) {
      (*this) << value;
    }
    return *this;
  }
  MatrixX<T>& operator<<(const T& value) {
    if (index_ < length_) {
      data_[index_++] = value;
    }

    return (*this);
  }
  MatrixX<T>& operator,(const T& value) { return (*this) << value; }

  MatrixX(const MatrixX<T>& other)
      : rows_(other.rows_), cols_(other.cols_), length_(other.length_), data_(other.data_) {}
  MatrixX(const VectorX<T>& other)
      : rows_(other.rows()), cols_(other.cols()), length_(other.length_), data_(other.data_) {}

  /// Call element [] operator
  // T& operator[](const size_t& index) {
  //   return data_[index];  // Adjusted for column-major order
  // }
  // const T& operator[](const size_t& index) const { return this->data_[index]; }

  T& operator()(const size_t& index) {
    return data_[index];  // Adjusted for column-major order
  }
  const T& operator()(const size_t& index) const { return this->data_[index]; }

  // T& operator[](const size_t& i, const size_t& j) { return data_[i * cols_ + j]; }
  // const T& operator[](const size_t& i, const size_t& j) const { return data_[i * cols_ + j]; }

  T& operator()(const size_t& i, const size_t& j) { return data_[i * cols_ + j]; }
  const T& operator()(const size_t& i, const size_t& j) const { return data_[i * cols_ + j]; }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t length() const { return cols() * rows(); }
  size_t numel() const { return length(); }
  std::vector<size_t> size() const { return std::vector<size_t>{rows, cols()}; }

  /// Slicing
  // get row
  MatrixX<T> row(size_t index) const {
    if (index >= rows_) {
      throw std::invalid_argument("Index out of range");
    }
    MatrixX<T> result(1, cols_);
    std::copy(data_.begin() + index * cols_, data_.begin() + (index + 1) * cols_,
              result.data_.begin());

    return result;
  }

  // get column
  const MatrixX<T> col(const size_t& index) const {
    if (index >= cols_) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(rows_, 1);  // Create the result matrix with the correct number of rows
    for (size_t i = 0; i < rows_; ++i) {
      result.data_[i] = data_[i * cols_ + index];
    }
    return result;
  }

  MatrixX<T> col(const std::vector<size_t>& indices) const {
    // Check if all indices are within range
    if (!std::all_of(indices.begin(), indices.end(), [&](size_t index) { return index < cols_; })) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(rows_, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      for (size_t j = 0; j < rows_; ++j) {
        result(j, i) = (*this)(j, indices[i]);
      }
    }

    return result;
  }

  MatrixX<T> row(const std::vector<size_t>& indices) const {
    // Check if all indices are within range
    if (!std::all_of(indices.begin(), indices.end(), [&](size_t index) { return index < rows_; })) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(indices.size(), cols_);

    // Copy each row from data_ to result_
    PARALLEL_FOR_COLLAPSE(1)
    std::for_each(indices.begin(), indices.end(), [&](size_t index) {
      std::copy(data_.begin() + index * cols_, data_.begin() + (index + 1) * cols_,
                result.data_.begin() + ((index - indices.front()) * cols_));
    });

    return result;
  }

  MatrixX<T> row(size_t start, size_t end) const {
    if (start >= rows_ || end >= rows_ || start > end) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(end - start + 1, cols_);

    // Generate each row in the specified range
    size_t rowIndex = start;
    std::generate(result.data_.begin(), result.data_.end(), [&]() {
      std::vector<T> rowValues(cols_);
      std::transform(data_.begin() + rowIndex * cols_, data_.begin() + (rowIndex + 1) * cols_,
                     rowValues.begin(), [](const T& elem) { return elem; });
      ++rowIndex;
      return rowValues;
    });

    return result;
  }

  // get column in the rang
  MatrixX<T> col(size_t start, size_t end) const {
    if (start >= cols_ || end >= cols_ || start > end) {
      throw std::invalid_argument("Index out of range");
    }
    MatrixX<T> result(rows_, end - start + 1);
    for (size_t i = 0; i < end - start + 1; ++i) {
      for (size_t j = 0; j < rows_; ++j) {
        result(j, i) = (*this)(j, start + i);
      }
    }
    return result;
  }

  // setRow
  MatrixX<T> setRow(const size_t& index, const MatrixX<T>& vec) {
    if (index >= rows_ || vec.length() != cols_) {
      throw std::invalid_argument("Index out of range");
    }

    std::transform(vec.data_.begin(), vec.data_.end(), data_.begin() + index * cols_,
                   [&](const T& elem) { return elem; });
    return *this;
  }

  void setCol(const size_t& index, const MatrixX<T>& vec) {
    if (index >= cols_ || vec.length() != rows_) {
      throw std::invalid_argument("Index out of range");
    }

    for (size_t i = 0; i < rows_; ++i) {
      data_[index + i * (cols_)] = vec.data_[i];
    }
  }

  /// Initialization

  // Random initialization
  void rand(T lowerBound = -1.0, T upperBound = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lowerBound, upperBound);

    std::transform(data_.begin(), data_.end(), data_.begin(),
                   [&](const T& /* ignored */) { return dis(gen); });
  }

  // Method to generate an zero matrix
  void zeros() {
    std::transform(data_.begin(), data_.end(), data_.begin(),
                   [&](const T& /* ignored */) { return 0; });
  }
  void ones() {
    std::transform(data_.begin(), data_.end(), data_.begin(),
                   [&](const T& /* ignored */) { return 1; });
  }

  // Method to generate an identity matrix
  void eye() {
    zeros();
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < std::min(rows_, cols_); ++i) {
      (*this)(i, i) = 1.0;
    }
  }

  /// Basic Operations
  // Transpose
  // const MatrixX<T>& transpose() {
  //   // Use std::transform to perform transpose
  //   std::transform(data_.begin(), data_.end(), data_.begin(), [this](const T& elem) {
  //     size_t index = (&elem - &data_[0]);
  //     return data_[(index % rows_) * cols_ + (index / rows_)];
  //   });

  //   std::swap(rows_, cols_);
  //   return *this;
  // }

  MatrixX<T> transpose() const {
    MatrixX<T> result(cols_, rows_);

    // Use std::transform to perform transpose
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(j, i) = (*this)(i, j);
      }
    }

    return result;
  }

  // Diagonal
  matrix::MatrixX<T> diag() const {
    if (rows_ == 1 || cols_ == 1) {
      matrix::MatrixX<T> result(std::max(rows_, cols_), std::max(rows_, cols_));
      for (size_t i = 0; i < std::max(rows_, cols_); ++i) {
        result(i, i) = (*this)(i);
      }
      return result;
    }
    matrix::MatrixX<T> result(std::min(rows_, cols_), 1);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < std::min(rows_, cols_); ++i) {
      result(i) = (*this)(i, i);
    }

    return result;
  }

  T sum() const {
    // Use std::accumulate to sum up all elements of the matrix
    return std::accumulate(data_.begin(), data_.end(), T(0));
  }

  MatrixX<T> sum(int axis) const {
    if (axis == 0) {
      MatrixX<T> result(1, cols_);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < cols_; ++i) {
        result(i) = col(i).sum();
      }
      return result;
    } else if (axis == 1) {
      MatrixX<T> result(rows_, 1);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < rows_; ++i) {
        result(i) = row(i).sum();
      }
      return result;
    } else {
      throw std::invalid_argument("Invalid axis");
    }
  }

  T mean() const { return sum() / (length_); }

  T prod() const {
    // Use std::accumulate to calculate the product of all elements of the matrix
    return std::accumulate(data_.begin(), data_.end(), T(1), std::multiplies<T>());
  }

  T trace() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for trace calculation");
    }

    // Use std::accumulate to calculate the trace of the matrix
    return std::accumulate(data_.begin(), data_.end(), T(0),
                           [&](const T& acc, const T& val) { return acc + val; });
  }

  T dot(const MatrixX<T>& other) const { return elementwiseProd(other).sum(); }

  T norm() const { return std::sqrt(dot(*this)); }

  MatrixX<T> norm(const int& axis) const {
    if (axis == 0) {
      MatrixX<T> result(1, cols_);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < cols_; ++i) {
        result(i) = col(i).norm();
      }
      return result;
    } else if (axis == 1) {
      MatrixX<T> result(rows_, 1);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < rows_; ++i) {
        result(i) = row(i).norm();
      }
      return result;
    } else {
      throw std::invalid_argument("Invalid axis");
    }
  }

  MatrixX<T> normalized() const {
    if (all([](T x) { return x == 0; })) {
      return *this;
    }
    return *this / norm();
  }

  // Method to find the minimum element and its index
  std::pair<size_t, T> minIndex() const {
    // Use std::min_element to find the minimum element in the data_
    auto min_it = std::min_element(data_.begin(), data_.end());

    // Get the index of the minimum element
    size_t index = std::distance(data_.begin(), min_it);

    // Return a pair containing the index and the minimum value
    return std::make_pair(index, *min_it);
  }

  // Method to find the maximum element and its index
  std::pair<size_t, T> maxIndex() const {
    // Use std::max_element to find the maximum element in the data_
    auto max_it = std::max_element(data_.begin(), data_.end());

    // Get the index of the maximum element
    size_t index = std::distance(data_.begin(), max_it);

    // Return a pair containing the index and the maximum value
    return std::make_pair(index, *max_it);
  }

  T max() const {
    // Use std::max_element to find the maximum element in the data_
    auto max_it = std::max_element(data_.begin(), data_.end());

    // Return the maximum element
    return *max_it;
  }

  T min() const {
    // Use std::max_element to find the maximum element in the data_
    auto min_it = std::min_element(data_.begin(), data_.end());

    // Return the maximum element
    return *min_it;
  }

  MatrixX<T> max(int axis) const {
    if (axis == 0) {
      MatrixX<T> result(1, cols_);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < cols_; ++i) {
        result(i) = col(i).max();
      }
      return result;
    } else if (axis == 1) {
      MatrixX<T> result(rows_, 1);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < rows_; ++i) {
        result(i) = row(i).max();
      }
      return result;
    } else {
      throw std::invalid_argument("Invalid axis");
    }
  }

  MatrixX<T> min(int axis) const {
    if (axis == 0) {
      MatrixX<T> result(1, cols_);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < cols_; ++i) {
        result(i) = col(i).min();
      }
      return result;
    } else if (axis == 1) {
      MatrixX<T> result(rows_, 1);
      PARALLEL_FOR_COLLAPSE(2)
      for (size_t i = 0; i < rows_; ++i) {
        result(i) = row(i).min();
      }
      return result;
    } else {
      throw std::invalid_argument("Invalid axis");
    }
  }

  /// Overloaded operator for scalar
  MatrixX<T> operator=(const MatrixX<T>& other) {
    if (this != &other) {
      data_ = other.data_;
      rows_ = other.rows_;
      cols_ = other.cols_;
      length_ = other.length_;
    }
    return *this;
  }

  MatrixX<T>& operator=(MatrixX<T>&& other) noexcept {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;
      length_ = other.length_;
    }
    return *this;
  }

  MatrixX<T> operator+(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < length_; ++i) {
      result(i) = data_[i] + scalar;
    }
    return result;
  }

  MatrixX<T> operator*(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < length_; ++i) {
      result(i) = data_[i] * scalar;
    }
    return result;
  }

  friend MatrixX<T> operator+(const T& scalar, const MatrixX<T>& matrix) { return matrix + scalar; }
  friend MatrixX<T> operator*(const T& scalar, const MatrixX<T>& matrix) { return matrix * scalar; }
  friend MatrixX<T> operator-(const T& scalar, const MatrixX<T>& matrix) {
    return -1 * matrix + scalar;
  }

  MatrixX<T> operator-(const T& scalar) const { return *this + (-scalar); }
  MatrixX<T> operator/(const T& scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero");
    }

    return (*this) * (1.0 / scalar);
  }

  MatrixX<T> operator*=(const T& scalar) { return *this = *this * scalar; }
  MatrixX<T> operator-=(const T& scalar) { return *this = *this - scalar; }
  MatrixX<T> operator/=(const T& scalar) { return *this = *this / scalar; }

  /// overloading operators for matrix
  MatrixX<T> operator+(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match.");
    }

    MatrixX<T> result(rows_, cols_);

    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   [](const T& a, const T& b) { return a + b; });

    return result;
  }

  // MatrixX multiplication
  MatrixX<T> operator*(const MatrixX<T>& other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument("MatrixX dimensions do not match for multiplication");
    }

// Use Strassen algorithm for large matrices
#if STARSSAN_ENABLED > 0
    if (std::max(rows_, cols_) > 500) {
      return strassen(other);
    }
#endif
    MatrixX<T> result(rows_, other.cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < other.cols_; ++j) {
        result(i, j) =
            std::inner_product(data_.begin() + i * cols_, data_.begin() + (i + 1) * cols_,
                               other.data_.begin() + j, T(0));
      }
    }

    return result;
  }
  MatrixX<T> operator-(const MatrixX<T>& other) const { return (*this) + (-1 * other); };
  MatrixX<T> operator/(const MatrixX<T>& other) { return *this * other.inv(); };

  MatrixX<T> operator+=(const MatrixX<T>& other) { return *this = *this + other; };
  MatrixX<T> operator-=(const MatrixX<T>& other) { return *this = *this - other; };
  MatrixX<T> operator*=(const MatrixX<T>& other) { return *this = *this * other; };

  friend MatrixX<T> operator/(const T& scalar, const MatrixX<T>& matrix) {
    return scalar * matrix.inv();
  }

  /// Apply element-wise function
  template <typename U>
  MatrixX<U> elementwise(std::function<U(const T&)> func) const {
    MatrixX<U> result(rows_, cols_);

    std::transform(data_.begin(), data_.end(), result.data_.begin(),
                   [&](const T& elem) { return func(elem); });

    return result;
  }

  /// self-elementwise function
  void selfElementwise(std::function<T(const T&)> func) {
    std::transform(data_.begin(), data_.end(), data_.begin(),
                   [&](const T& elem) { return func(elem); });
  }

  template <typename U>
  MatrixX<U> elementwise(const MatrixX<T>& other, std::function<U(const T&, const T&)> func) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for element-wise operation");
    }

    MatrixX<U> result(rows_, cols_);

    // Use std::transform to perform element-wise operation
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   [&](const T& a, const T& b) { return func(a, b); });

    return result;
  }

  bool all(std::function<bool(const T&)> func) const {
    return std::all_of(data_.begin(), data_.end(), func);
  }

  bool all(const MatrixX<T>& other, std::function<bool(const T&, const T&)> func) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for element-wise comparison");
    }

    // Use std::mismatch to find the first element where the matrices differ
    auto mismatchPair = std::mismatch(data_.begin(), data_.end(), other.data_.begin(), func);

    // If there's no mismatch, return true
    return mismatchPair.first == data_.end();
  }

  bool any(std::function<bool(const T&)> func) const {
    return std::any_of(data_.begin(), data_.end(), func);
  }
  bool any(const MatrixX<T>& other, std::function<bool(const T&, const T&)> func) {
    return !all(other, [&](const T& x, const T& y) { return !func(x, y); });
  }

  MatrixX<T> elementwiseProd(const MatrixX<T>& other) const {
    return elementwise<T>(other, [](T x, T y) { return x * y; });
  }

  MatrixX<T> elementwiseDiv(const MatrixX<T>& other) const {
    return elementwise<T>(other, [](T x, T y) {
      if (y == 0) {
        throw std::invalid_argument("Division by zero");
      }
      return x / y;
    });
  }

  /// Apply a function to each element of the matrix
  MatrixX<T> abs() const {
    return elementwise<T>([](const T& x) { return std::abs(x); });
  }

  MatrixX<T> sin() const {
    return elementwise<T>([](T x) { return std::sin(x); });
  }

  MatrixX<T> cos() const {
    return elementwise<T>([](T x) { return std::cos(x); });
  }

  MatrixX<T> tan() const {
    return elementwise<T>([](T x) { return std::tan(x); });
  }

  MatrixX<T> asin() const {
    return elementwise<T>([](T x) { return std::asin(x); });
  }

  MatrixX<T> acos() const {
    return elementwise<T>([](T x) { return std::acos(x); });
  }

  MatrixX<T> atan() const {
    return elementwise<T>([](T x) { return std::atan(x); });
  }

  MatrixX<T> tanh() const {
    return elementwise<T>([](T x) { return std::tanh(x); });
  }

  // sigmoid function
  MatrixX<T> sigmoid() const {
    return elementwise<T>([](T x) { return 1.0 / (1.0 + std::exp(-x)); });
  }

  MatrixX<T> relu() const {
    return elementwise<T>([](T x) { return (x > 0) ? x : 0; });
  }

  //  square root function
  const MatrixX<T> sqrt() const {
    return elementwise<T>([](T x) { return std::sqrt(x); });
  }

  //  square  function
  MatrixX<T> pow2() const {
    return elementwise<T>([](T x) { return x * x; });
  }

  //  exponential function
  MatrixX<T> exp() const {
    return elementwise<T>([](T x) { return std::exp(x); });
  }

  //  log function
  MatrixX<T> log() const {
    return elementwise<T>([](T x) { return std::log(x); });
  }

  // square function
  MatrixX<T> square() const {
    return elementwise<T>([](T x) { return x * x; });
  }

  // softmax function
  MatrixX<T> softmax() const {
    MatrixX<T> result(rows_, cols_);

    T s = elementwise<T>([](T x) { return std::exp(x); }).sum();
    return elementwise<T>([s](T x) { return std::exp(x) / s; });
  }

  // ceil function
  MatrixX<T> ceil() const {
    return elementwise<T>([](const T& x) { return std::ceil(x); });
  }

  // floor function
  MatrixX<T> floor() const {
    return elementwise<T>([](const T& x) { return std::floor(x); });
  }

  // round the matrix to zero if smaller than a threshold
  MatrixX<T> round(const T& threshold = 1e-6) const {
    return elementwise<T>([threshold](const T& x) { return (std::abs(x) < threshold) ? 0 : x; });
  }

  void selfRound(const T& threshold = 1e-6) const {
    selfElementwise<T>([threshold](const T& x) { return (std::abs(x) < threshold) ? 0 : x; });
  }

  // bound a matrix
  MatrixX<T> bound(const T& lowerBound, const T& upperBound) const {
    return elementwise<T>([lowerBound, upperBound](const T& x) {
      return std::min(std::max(x, lowerBound), upperBound);
    });
  }

  // clip a matrix
  MatrixX<T> clip(const T& lowerBound, const T& upperBound) const {
    return elementwise<T>([lowerBound, upperBound](const T& x) {
      return (x < lowerBound) ? lowerBound : (x > upperBound) ? upperBound : x;
    });
  }

  // clip a matrix
  MatrixX<T> clip(const std::pair<T, T>& bounds) const { return clip(bounds.first, bounds.second); }

  /// Inverse
  // MatrixX inv using Gaussian elimination (without pivoting)
  MatrixX<T> inv() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for inv calculation");
    }

    size_t n = rows_;

    // Augmenting the matrix with identity matrix of the same size
    MatrixX<T> augmented(n, 2 * n);
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        augmented(i, j) = (*this)(i, j);
        augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
      }
    }

    // Forward elimination
    PARALLEL_FOR_COLLAPSE(3)
    for (size_t i = 0; i < n; ++i) {
      // Divide the row by the diagonal element to make it 1
      T pivot = augmented(i, i);
      if (pivot == 0) {
        throw std::runtime_error("MatrixX is singular, inv does not exist");
      }
      for (size_t j = 0; j < 2 * n; ++j) {
        augmented(i, j) /= pivot;
      }

      // Eliminate non-zero elements below the diagonal
      for (size_t k = 0; k < n; ++k) {
        if (k != i) {
          T factor = augmented(k, i);
          for (size_t j = 0; j < 2 * n; ++j) {
            augmented(k, j) -= factor * augmented(i, j);
          }
        }
      }
    }

    // Extracting the inv from the augmented matrix
    MatrixX<T> result(n, n);
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result(i, j) = augmented(i, j + n);
      }
    }

    return result;
  }

  MatrixX<T> submatrix(const size_t& nx, const size_t& ny) const {
    if (nx >= rows_ || ny >= cols_) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(rows_ - 1, cols_ - 1);
    size_t newRow = 0;
    for (size_t i = 0; i < rows_; ++i) {
      if (i == nx) {
        continue;  // Skip the row at index nx
      }
      size_t newCol = 0;
      for (size_t j = 0; j < cols_; ++j) {
        if (j == ny) {
          continue;  // Skip the column at index ny
        }
        result(newRow, newCol) = (*this)(i, j);
        ++newCol;
      }
      ++newRow;
    }
    return result;
  }
  static void DivideAndConquer(std::vector<T>& vec,
                               std::function<void(std::vector<T>&)> operation) {
    // Base case: If the vector is empty or has only one element, apply the operation directly
    if (vec.size() <= 1) {
      operation(vec);
      return;
    }

    // Split the vector into two halves
    size_t mid = vec.size() / 2;
    std::vector<T> left(vec.begin(), vec.begin() + mid);
    std::vector<T> right(vec.begin() + mid, vec.end());

    // Recursively apply the operation on each half
    DivideAndConquer(left, operation);
    DivideAndConquer(right, operation);
  }

  // cofactor
  T cofactor(const size_t& nx, const size_t& ny) const {
    return submatrix(nx, ny).det() * ((nx + ny) % 2 == 0 ? 1 : -1);
  }

  // determinant
  T det() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for determinant calculation");
    }

    if (rows_ == 1) {
      return (*this)(0, 0);
    }

    T result = 0;
    for (size_t i = 0; i < rows_; ++i) {
      result += (*this)(0, i) * cofactor(0, i);
    }
    return result;
  }

  // adjoint
  MatrixX<T> adjoint() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for adjoint calculation");
    }

    MatrixX<T> result(rows_, cols_);
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = cofactor(i, j);
      }
    }

    return result.transpose();
  }

  // lu decomposition using pivot matrix
  std::tuple<MatrixX<T>, MatrixX<T>, MatrixX<T>> lu() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for LU decomposition");
    }

    MatrixX<T> L(rows_, cols_);
    MatrixX<T> U(rows_, cols_);
    MatrixX<T> P(rows_, cols_);
    P.eye();

    // Copy the matrix to U
    U = *this;

    // Perform the LU decomposition
    for (size_t i = 0; i < rows_; ++i) {
      // Partial pivoting: find pivot row
      size_t rp = i;
      T max_val = std::abs(U(i, i));
      for (size_t k = i + 1; k < rows_; k++) {
        if (std::abs(U(k, i)) > max_val) {
          max_val = std::abs(U(k, i));
          rp = k;
        }
      }
      // Swap rows
      if (rp != i) {
        U.swapRows(i, rp);
        P.swapRows(i, rp);
        L.swapRows(i, rp);
      }

      // Perform elimination
      for (size_t j = i + 1; j < rows_; ++j) {
        T factor = U(j, i) / U(i, i);
        for (size_t k = i; k < rows_; ++k) {
          U(j, k) -= factor * U(i, k);
        }
        U(j, i) = factor;
      }
    }

    // Extract L
    for (size_t i = 0; i < rows_; ++i) {
      L(i, i) = 1;
      for (size_t j = 0; j < i; ++j) {
        L(i, j) = U(i, j);
        U(i, j) = 0;
      }
    }

    return std::make_tuple(L, U, P);
  }

  MatrixX<T> pinv(const MatrixX<T>& matrix, const T& tol = std::numeric_limits<T>::epsilon()) {
    // Calculate the singular value decomposition (SVD)
    auto svd = matrix.svd();
    const auto& U = svd.first;
    const auto& sigma = svd.second;
    const auto& Vt = svd.third;

    // Calculate the reciprocal of non-zero singular values
    MatrixX<T> invSigma(matrix.cols(), matrix.rows());
    invSigma.zeros();
    for (size_t i = 0; i < sigma.rows(); ++i) {
      if (std::abs(sigma(i)) > tol) {
        invSigma(i, i) = 1.0 / sigma(i);
      }
    }

    // Calculate the pseudoinverse
    return Vt.transpose() * invSigma * U.transpose();
  }

  // Swap two rows without using std::swap
  void swapRows(const size_t& i, const size_t& j) {
    if (i == j) {
      return;
    }

    for (size_t k = 0; k < cols_; ++k) {
      std::swap(data_[i * cols_ + k], data_[j * cols_ + k]);
    }
  }

  // Swap two columns without using std::swap
  void swapCols(const size_t& i, const size_t& j) {
    if (i == j) {
      return;
    }

    for (size_t k = 0; k < rows_; ++k) {
      std::swap(data_[i + k * cols_], data_[j + k * cols_]);
    }
  }

  // gaussianElimination
  static MatrixX<T> GaussianElimination(const MatrixX<T>& matrix) {
    MatrixX<T> result = matrix;
    size_t rows_ = matrix.rows();

    for (size_t r = 0; r < rows_; ++r) {
      // Find the pivot row
      auto maxRowIter =
          std::max_element(result.data_.begin() + r * rows_, result.data_.end(),
                           [&](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
      size_t rp = std::distance(result.data_.begin(), maxRowIter) / rows_;

      // Swap rows
      if (rp != r) {
        result.swapRows(r, rp);
      }

      // Perform elimination
      for (size_t j = r + 1; j < rows_; ++j) {
        T factor = result(j, r) / result(r, r);
        for (size_t k = r; k < rows_; ++k) {
          result(j, k) -= factor * result(r, k);
        }
      }
    }

    return result;
  }

  // qr decomposition
  std::pair<MatrixX<T>, MatrixX<T>> qr() const {
    const size_t m = rows_;
    const size_t n = cols_;
    auto R = MatrixX<T>(n, n);

    R.zeros();
    auto Q = *this;

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t k = 0; k < n; ++k) {
      for (size_t i = 0; i < k; ++i) {
        R(i, k) = Q.col(i).dot(Q.col(k));
      }
      for (size_t i = 0; i < k; ++i) {
        Q.setCol(k, Q.col(k) - R(i, k) * Q.col(i));
      }
      R(k, k) = Q.col(k).norm();
      Q.setCol(k, Q.col(k) / R(k, k));
    }

    return std::make_pair(Q, R);
  }

  // remove small values
  MatrixX<T> removeEps(const T& tol = std::numeric_limits<T>::epsilon()) const {
    return elementwise<T>([tol](T x) { return (std::abs(x) < tol) ? 0 : x; });
  }

  static std::pair<MatrixX<T>, T> Jacobi(const T& alpha, const T& beta, const T& gamma) {
    T c, s, t;

    MatrixX<T> G(2, 2);
    G.eye();

    if (beta == 0.0) {
      // If beta is zero, no rotation is needed, return the identity matrix
      std::pair<MatrixX<T>, T>(G, 0);
    }

    // Compute tau
    T tau = (gamma - alpha) / (2 * beta);

    // Compute t
    if (tau >= 0) {
      t = 1 / (tau + std::sqrt(1 + tau * tau));
    } else {
      t = -1 / (-tau + std::sqrt(1 + tau * tau));
    }

    // Compute cosine and sine
    c = 1 / std::sqrt(1 + t * t);
    s = t * c;

    // Fill in the Jacobi transformation matrix
    G(0, 0) = c;
    G(0, 1) = s;
    G(1, 0) = -s;
    G(1, 1) = c;

    return std::pair<MatrixX<T>, T>(G, t);
  }

  matrix::MatrixX<T> eig() {
    MatrixX<T> A = *this;
    MatrixX<T> sigma = square().sum(1);
    T eps = 10 * std::numeric_limits<T>::epsilon();
    T tol_sigma = eps * norm();
    while (true) {
      size_t rots = 0;
      for (size_t p = 0; p < cols_ - 1; p++) {
        for (size_t q = p + 1; q < cols_; q++) {
          T beta = (A.col(p).transpose() * A.col(q))(0);
          if (sigma(p) * sigma(q) > tol_sigma &&
              std::abs(beta) >= eps * std::sqrt(sigma(p) * sigma(q))) {
            rots++;
            auto res = Jacobi(sigma(p), beta, sigma(q));  // jacobi rotation matrix
            MatrixX<T> G = res.first;
            T t = res.second;
            // update eigenvalues
            sigma(p) -= beta * t;
            sigma(q) += beta * t;
            auto temp = ConcatCols(A.col(p), A.col(q)) * G;
            A.setCol(p, temp.col(0));
            A.setCol(q, temp.col(1));
          }
        }
      }
      if (rots == 0)  // Convergence reached
        break;
    }
    return sigma.sqrt();
  }

  std::tuple<MatrixX<T>, MatrixX<T>, MatrixX<T>> svd() {  // returns U,E,VT where U*E*VT = Object
    MatrixX<T> A = *this;
    MatrixX<T> U(rows_, rows_);
    MatrixX<T> V(cols_, cols_);
    MatrixX<T> sigma = square().sum(1);

    V.eye();

    T eps = 10 * std::numeric_limits<T>::epsilon();
    T tol_sigma = eps * norm();
    while (true) {  // iterate until all (p,q) pairs give less error than tolerance
      size_t rots = 0;
      for (size_t p = 0; p < cols_ - 1; p++) {
        for (size_t q = p + 1; q < cols_; q++) {
          T beta = (A.col(p).transpose() * A.col(q))(0);
          if (sigma(p) * sigma(q) > tol_sigma &&
              std::abs(beta) >= eps * std::sqrt(sigma(p) * sigma(q))) {
            rots++;
            auto res = Jacobi(sigma(p), beta, sigma(q));  // jacobi rotation matrix
            MatrixX<T> G = res.first;
            T t = res.second;
            // update eigenvalues
            sigma(p) -= beta * t;
            sigma(q) += beta * t;
            auto temp = ConcatCols(A.col(p), A.col(q)) * G;
            A.setCol(p, temp.col(0));
            A.setCol(q, temp.col(1));
            MatrixX<T> temp2 = ConcatCols(V.col(p), V.col(q)) * G;
            V.setCol(p, temp.col(0));
            V.setCol(q, temp.col(1));
          }
        }
      }
      if (rots == 0)  // Convergence reached
        break;
    }
    // post processing w.r.t. indices (eigenvalues should appear in descending order)
    MatrixX<T> cV = V;
    std::vector<std::pair<size_t, T>> ind(sigma.cols());

    for (size_t i = 0; i < sigma.cols(); i++) {
      ind[i] = std::make_pair(i, sigma(i));
    }

    sort(ind.begin(), ind.end(),
         [](std::pair<size_t, T>& p1,
            std::pair<size_t, T>& p2) {  // utility function to compare pairs
           return p1.second > p2.second;
         });

    // Construct U and V
    for (size_t i = 0; i < U.rows(); i++) {
      for (size_t j = 0; j < ind.size(); j++) {
        sigma(j) = ind[j].second;
        U(i, j) = A(i, ind[j].first);
      }
    }
    for (size_t i = 0; i < cV.rows(); i++) {
      for (size_t j = 0; j < ind.size(); j++) {
        V(i, j) = cV(i, ind[j].first);
      }
    }
    // find eigenvalues
    for (size_t k = 0; k < cols(); k++) {
      if (sigma(k) == 0) {
        for (size_t j = k + 1; j < cols(); j++) {
          sigma(j) = 0;
        }
      }
      for (size_t i = 0; i < rows(); i++) {
        U(i, k) /= sigma(k);
      }
    }

    return std::tuple<MatrixX<T>, MatrixX<T>, MatrixX<T>>(U, sigma.sqrt().diag(), V.transpose());
  }

  std::pair<MatrixX<T>, MatrixX<T>> eigen(const T& tol = std::numeric_limits<T>::epsilon() * 100,
                                          const size_t& max_iter = 999) const {
    auto D = *this;
    MatrixX<T> V(rows_, rows_);
    V.eye();
    auto oldD = D;
    oldD.zeros();

    size_t i = 0;
    T error = std::numeric_limits<T>::max();

    while (error > tol && i < max_iter) {
      MatrixX<T> Q, R;
      std::tie(Q, R) = D.qr();
      D = R * Q;
      V = V * Q;
      i++;
    }

    return std::pair<MatrixX<T>, MatrixX<T>>(D, V);
  }

  /// binary operations
  bool operator==(const MatrixX<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < length_; ++i) {
      if (data_[i] != other(i)) {
        return false;
      }
    }
    return true;
  }

  // Boolean greater than operator
  MatrixX<T> operator>(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x > scalar; });
  }

  // Boolean less than operator
  MatrixX<T> operator<(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x < scalar; });
  }

  // Boolean greater than or equal to operator
  MatrixX<T> operator>=(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x >= scalar; });
  }

  // Boolean less than or equal to operator
  MatrixX<T> operator<=(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x <= scalar; });
  }

  // Boolean equal to operator
  MatrixX<T> operator==(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x == scalar; });
  }

  // Boolean not equal to operator
  MatrixX<T> operator!=(const T& scalar) const {
    return elementwise<T>([scalar](T x) { return x != scalar; });
  }

  // Boolean greater than operator
  MatrixX<T> operator>(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x > y; });
  }

  // Boolean less than operator
  MatrixX<T> operator<(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x < y; });
  }

  // Boolean greater than or equal to operator
  MatrixX<T> operator>=(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x >= y; });
  }

  // Boolean less than or equal to operator
  MatrixX<T> operator<=(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x <= y; });
  }

  // Boolean equal to operator
  MatrixX<T> operator==(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x == y; });
  }

  // Boolean not equal to operator
  MatrixX<T> operator!=(const MatrixX<T>& other) const {
    return elementwise<T>(other, [&](const T& x, const T& y) { return x != y; });
  }

  // do column-wise sum
  MatrixX<T> rowPlus(const MatrixX<T>& vec = 0) const {
    MatrixX<T> result = *this;
    if (vec.cols() != cols_ && vec.rows() != 1) {
      throw std::invalid_argument("Invalid vector dimensions for row-wise sum");
    }
    for (size_t i = 0; i < length_; i++) {
      result.data_[i] += vec.data_[i % vec.length_];
    }

    return result;
  }

  // do column-wise sum
  MatrixX<T> colPlus(const MatrixX<T>& vec = 0) const {
    MatrixX<T> result = *this;
    if (vec.rows() != rows_ && vec.cols() != 1) {
      throw std::invalid_argument("Invalid vector dimensions for column-wise sum");
    }
    for (size_t i = 0; i < length_; i++) {
      result.data_[i] += vec.data_[i / cols_];
    }

    return result;
  }

  /// Casting
  // cast to another type
  template <typename U>
  MatrixX<U> cast() const {
    MatrixX<U> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < length_; ++i) {
      result(i) = static_cast<U>(data_[i]);
    }

    return result;
  }

  // cast to std::vector
  std::vector<T> toVector() const {
    std::vector<T> result(length_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < length_; ++i) {
      result(i) = data_[i];
    }

    return result;
  }

  /// Resize
  // resize the matrix
  void resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(length_, 0.0);
  }

  // column concetenate two matrices
  static MatrixX<T> ConcatCols(const MatrixX<T>& A, const MatrixX<T>& B) {
    if (B.rows_ != A.rows_) {
      throw std::invalid_argument("MatrixX dimensions do not match for column concatenation");
    }

    MatrixX<T> result(A.rows_, A.cols_ + B.cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < A.rows_; ++i) {
      for (size_t j = 0; j < A.cols_; ++j) {
        result(i, j) = A(i, j);
      }
      for (size_t j = 0; j < B.cols_; ++j) {
        result(i, A.cols_ + j) = B(i, j);
      }
    }

    return result;
  }

  // row concetenate two matrices
  static MatrixX<T> ConcatRows(const MatrixX<T>& A, const MatrixX<T>& B) {
    if (B.cols_ != A.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for row concatenation");
    }

    MatrixX<T> result(A.rows_ + B.rows_, A.cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < A.rows_; ++i) {
      for (size_t j = 0; j < A.cols_; ++j) {
        result(i, j) = A(i, j);
      }
    }
    for (size_t i = 0; i < B.rows_; ++i) {
      for (size_t j = 0; j < B.cols_; ++j) {
        result(A.rows_ + i, j) = B(i, j);
      }
    }

    return result;
  }

  // concentration of two matrices by axis
  static MatrixX<T> Concat(const MatrixX<T>& A, const MatrixX<T>& B, int axis) {
    if (axis == 0) {
      return ConcatRows(A, B);
    } else if (axis == 1) {
      return ConcatCols(A, B);
    } else {
      throw std::invalid_argument("Invalid axis");
    }
  }

  // reshape the matrix
  MatrixX<T> reshape(const size_t& rows, const size_t& cols) const {
    if (rows * cols != length_) {
      throw std::invalid_argument("New shape must be compatible with the number of elements");
    }

    MatrixX<T> result(rows, cols, data_);

    return result;
  }

  // get block of matrix
  MatrixX<T> block(const size_t& rowStart, const size_t& rowEnd, const size_t& colStart,
                   const size_t& colEnd) const {
    if (rowStart > rows_ || rowEnd > rows_ || colStart > cols_ || colEnd > cols_) {
      throw std::invalid_argument("Invalid block indices");
    }

    size_t size_x = rowStart > rowEnd ? rowStart - rowEnd : rowEnd - rowStart;
    size_t size_y = colStart > colEnd ? colStart - colEnd : colEnd - colStart;

    MatrixX<T> result(size_x, size_y);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < size_x; ++i) {
      for (size_t j = 0; j < size_y; ++j) {
        result(i, j) = (*this)(rowStart + i, colStart + j);
      }
    }

    return result;
  }

  // set block of matrix
  void setBlock(const size_t& rowStart, const size_t& colStart, const MatrixX<T>& block) {
    if (rowStart + block.rows() > rows_ || colStart + block.cols() > cols_) {
      throw std::invalid_argument("Block does not fit in the matrix");
    }

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < block.rows(); ++i) {
      for (size_t j = 0; j < block.cols(); ++j) {
        (*this)(rowStart + i, colStart + j) = block(i, j);
      }
    }
  }

  // get raw data
  T* data() { return data_.data(); }

  operator VectorX<T>() const {
    VectorX<T> result;
    if (cols() == 1 || rows() == 1) {
      result = VectorX<T>(rows(), 0, cols());
      for (size_t i = 0; i < length(); ++i) {
        result(i) = (*this)(i);
      }
    } else {
      throw std::invalid_argument("Matrix must have only one column or row to be cast to a vector");
    }

    return result;
  };

  /// Printing
  void print(const std::string& header = "\nA=", const std::string& footer = "\n") {
    std::cout << header.c_str() << *this << footer.c_str();
  }
  friend std::ostream& operator<<(std::ostream& os, const MatrixX<T>& matrix) {
    os << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < matrix.rows(); ++i) {
      for (size_t j = 0; j < matrix.cols(); ++j) {
        os << matrix(i, j) << ", ";
      }
      if (i < matrix.rows() - 1) {
        os << std::endl;
      }
    }
    return os;
  }

  /// Saving Data
  void save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for writing: " << filename << std::endl;
      return;
    }

    save(file);

    file.close();
    std::cout << "The matrix is saved successfully." << filename << std::endl;
  }

  void save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&length_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(data_.data()), sizeof(T) * length_);
  }

  /// Loading Data
  void load(const std::string& filename) {
    std::cout << "Loading matrix from file" << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Failed to open file for reading: " << filename << std::endl;
      return;
    }

    load(file);

    file.close();
    std::cout << "The matrix is loaded successfully." << filename << std::endl;
  }

  void load(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&rows_), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols_), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&length_), sizeof(size_t));
    data_.resize(length_);
    file.read(reinterpret_cast<char*>(data_.data()), sizeof(T) * length_);
  }

  static MatrixX<T> Load(const std::string& filename) {
    MatrixX<T> result;
    result.load(filename);
    return result;
  }

  // Strassen's algorithm for matrix multiplication
  template <typename T>
  MatrixX<T> strassen(const MatrixX<T>& other) const {
    auto A = *this;
    auto B = other;

    // Check dimensions
    if (A.cols_ != B.rows_) {
      throw std::invalid_argument("MatrixX dimensions do not match for Strassen multiplication");
    }

    // Padding to make dimensions even
    size_t m = A.rows_;
    size_t n = A.cols_;
    size_t s = B.cols_;
    size_t n2 = std::max({m, n, s}) - 1;
    n2 = n2 | (n2 >> 1);
    n2 = n2 | (n2 >> 2);
    n2 = n2 | (n2 >> 4);
    n2 = n2 | (n2 >> 8);
    n2 = n2 | (n2 >> 16);
    n2 += 1;

    size_t pm = 0;
    size_t ps = 0;
    if (n2 > m || n2 > n || n2 > s) {
      A = MatrixX<T>::ConcatCols(MatrixX<T>::ConcatRows(A, MatrixX<T>(n2 - m, n)),
                                 MatrixX<T>(n2, n2 - n));
      B = MatrixX<T>::ConcatCols(MatrixX<T>::ConcatRows(B, MatrixX<T>(n2 - n, s)),
                                 MatrixX<T>(n2, n2 - s));

      pm = n2 - m;
      ps = n2 - s;

      m = n2;
      n = n2;
      s = n2;
    }

    // Base case
    if (m == 1 && n == 1 && s == 1) {
      return MatrixX<T>({{A(0, 0) * B(0, 0)}});
    }

    // Partition matrices
    size_t h = m / 2;

    MatrixX<T> A11 = A.block(0, h, 0, h);
    MatrixX<T> A12 = A.block(0, h, h, m);
    MatrixX<T> A21 = A.block(h, m, 0, h);
    MatrixX<T> A22 = A.block(h, m, h, m);

    MatrixX<T> B11 = B.block(0, h, 0, h);
    MatrixX<T> B12 = B.block(0, h, h, m);
    MatrixX<T> B21 = B.block(h, m, 0, h);
    MatrixX<T> B22 = B.block(h, m, h, m);

    // Strassen's recursive calls
    MatrixX<T> M1, M2, M3, M4, M5, M6, M7;
#pragma omp parallel sections
    {
#pragma omp section
      M1 = (A11 + A22) * (B11 + B22);
#pragma omp section
      M2 = (A21 + A22) * (B11);
#pragma omp section
      M3 = A11 * (B12 - B22);
#pragma omp section
      M4 = A22 * (B21 - B11);
#pragma omp section
      M5 = (A11 + A12) * (B22);
#pragma omp section
      M6 = (A21 - A11) * (B11 + B12);
#pragma omp section
      M7 = (A12 - A22) * (B21 + B22);
    }

    // Combine results
    MatrixX<T> C11 = M1 + M4 - M5 + M7;
    MatrixX<T> C12 = M3 + M5;
    MatrixX<T> C21 = M2 + M4;
    MatrixX<T> C22 = M1 - M2 + M3 + M6;

    // Remove padding and return
    return MatrixX<T>::ConcatCols(MatrixX<T>::ConcatRows(C11, C21),
                                  MatrixX<T>::ConcatRows(C12, C22))
        .block(0, m - pm, 0, s - ps);
  }

 protected:
  std::vector<T> data_;
  size_t rows_{0};
  size_t cols_{0};
  size_t length_{0};
  size_t index_{0};
};

template <typename T>
class eyeX : public MatrixX<T> {
 public:
  eyeX(const size_t& n, const size_t& m) : MatrixX<T>(n, m) { eye(); }
  eyeX(const size_t& n) : MatrixX<T>(n, n) { eye(); }
};

template <typename T>
class zerosX : public MatrixX<T> {
 public:
  zerosX(const size_t& rows, const size_t& cols) : MatrixX<T>(rows, cols) { zeros(); }
  zerosX(const size_t& n) : MatrixX<T>(n, n) { zeros(); }
};

template <typename T>
class onesX : public MatrixX<T> {
 public:
  onesX(const size_t& rows, const size_t& cols) : MatrixX<T>(rows, cols) { ones(); }
  onesX(const size_t& n) : MatrixX<T>(n, n) { ones(); }
};

template <typename T>
class randX : public MatrixX<T> {
 public:
  randX(const size_t& rows, const size_t& cols, const T& lower_bound = -1, const T& upper_bound = 1)
      : MatrixX<T>(rows, cols) {
    rand(lower_bound, upper_bound);
  }
  randX(const size_t& n, const T& lower_bound = -1, const T& upper_bound = 1) : MatrixX<T>(n, n) {
    rand(lower_bound, upper_bound);
  }
};

typedef MatrixX<double> Matrix;
typedef MatrixX<float> Matrixf;
typedef MatrixX<math::Complex> Matrixc;

typedef eyeX<double> eye;
typedef eyeX<float> eyef;
typedef zerosX<double> zeros;
typedef zerosX<float> zerosf;
typedef onesX<double> ones;
typedef onesX<float> onesf;
typedef randX<double> rand;
typedef randX<float> randf;

}  // namespace matrix
#include "matrix/special_matrix.hpp"
