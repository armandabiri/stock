#pragma once

#include <tuple>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <functional>
#include <cmath>
#include <omp.h>
#include <iomanip>  // for std::setw

#include "matrix/Complex.hpp"
#include "matrix/utils.hpp"

#define PARALLEL_ENABLED 1  // Set to 1 to enable parallel computing, 0 to disable
#define STARSSAN_ENABLED 1  // Set to 1 to enable Strassen algorithm, 0 to disable

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
  MatrixX() : rows_(0), cols_(0) {}
  MatrixX(const size_t& rows, const size_t& cols) : rows_(rows), cols_(cols) {
    data_.resize(rows_ * cols_);
  };
  MatrixX(const size_t& rows, const size_t& cols, T value) : rows_(rows), cols_(cols) {
    data_.resize(rows_ * cols_, value);
  }

  MatrixX(const size_t& rows, const size_t& cols, const std::initializer_list<T>& list)
      : rows_(rows), cols_(cols) {
    if (list.size() != rows * cols) {
      throw std::invalid_argument("Initializer list size does not match matrix size");
    }
    data_.resize(rows_ * cols_);
    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      data_[i++] = *it;
    }
  }

  MatrixX(const std::initializer_list<T>& list) : {
    rows_ = list.size();
    cols_ = 1;
    if (list.size() != rows_ * cols_) {
      throw std::invalid_argument("Initializer list size does not match matrix size");
    }
    data_.resize(rows_ * cols_);
    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      data_[i++] = *it;
    }
  }

  MatrixX(const std::initializer_list<std::initializer_list<T>>& list) {
    rows_ = list.size();
    cols_ = list.begin()->size();

    data_.resize(rows_ * cols_);
    size_t i = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
      if (it->size() != cols_) {
        throw std::invalid_argument("Initializer list size does not match matrix size");
      }
      for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
        data_[i++] = *it2;
        ;
      }
    }
  }

  MatrixX(const MatrixX<T>& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}
  MatrixX(const VectorX<T>& other)
      : rows_(other.rows()), cols_(other.cols()), data_(other.data()) {}

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
    for (size_t i = 0; i < cols_; ++i) {
      result(i) = (*this)(index, i);
    }
    return result;
  }

  // get multiple cols by give the index vector
  MatrixX<T> col(const std::vector<size_t>& indices) const {
    MatrixX<T> result(rows_, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= cols_) {
        throw std::invalid_argument("Index out of range");
      }
      for (size_t j = 0; j < rows_; ++j) {
        result(j, i) = (*this)(j, indices[i]);
      }
    }
    return result;
  }

  // get multiple cols by give the index vector
  MatrixX<T> row(const std::vector<size_t>& indices) const {
    MatrixX<T> result(indices.size(), cols_);
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= rows_) {
        throw std::invalid_argument("Index out of range");
      }
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = (*this)(indices[i], j);
      }
    }
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

  // get row in the rang
  MatrixX<T> row(size_t start, size_t end) const {
    if (start >= rows_ || end >= rows_ || start > end) {
      throw std::invalid_argument("Index out of range");
    }
    MatrixX<T> result(end - start + 1, cols_);
    for (size_t i = 0; i < end - start + 1; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = (*this)(start + i, j);
      }
    }
    return result;
  }

  // get column
  const MatrixX<T> col(const size_t& index) const {
    if (index >= cols_) {
      throw std::invalid_argument("Index out of range");
    }

    MatrixX<T> result(rows_, 1);
    for (size_t i = 0; i < rows_; ++i) {
      result(i) = (*this)(i, index);
    }
    return result;
  }

  void setCol(const size_t& index, const MatrixX<T>& vec) {
    if (index >= cols_ || vec.length() != rows_) {
      throw std::invalid_argument("Index out of range");
    }

    for (size_t i = 0; i < rows_; ++i) {
      (*this)(i, index) = vec(i);
    }
  }

  // setRow
  MatrixX<T> setRow(const size_t& index, const MatrixX<T>& vec) {
    if (index >= rows_ || vec.length() != cols_) {
      throw std::invalid_argument("Index out of range");
    }

    for (size_t i = 0; i < cols_; ++i) {
      (*this)(index, i) = vec(i);
    }
    return *this;
  }

  /// Initialization

  // Random initialization
  void rand(T lowerBound = -1.0, T upperBound = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lowerBound, upperBound);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      data_[i] = dis(gen);
    }
  }

  // Method to generate an zero matrix
  void zeros() {
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      data_[i] = 0;
    }
  }
  void ones() {
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      data_[i] = 1;
    }
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
  MatrixX<T> transpose() const {
    MatrixX<T> result(cols_, rows_);
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
    T result = 0.0;
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result += data_[i];
    }
    return result;
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

  T mean() const { return sum() / (rows_ * cols_); }

  T prod() const {
    T result = 1.0;

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result *= data_[i];
    }
    return result;
  }

  T trace() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for trace calculation");
    }

    T result = 0.0;

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < rows_; ++i) {
      result += (*this)(i, i);
    }
    return result;
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

  T max() const {
    T result = data_[0];

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 1; i < numel(); ++i) {
      if (data_[i] > result) {
        result = data_[i];
      }
    }
    return result;
  }

  T min() const {
    T result = data_[0];

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 1; i < numel(); ++i) {
      if (data_[i] < result) {
        result = data_[i];
      }
    }
    return result;
  }

  /// Overloaded operator for scalar
  MatrixX<T> operator=(const MatrixX<T>& other) {
    if (this != &other) {
      data_ = other.data_;
      rows_ = other.rows_;
      cols_ = other.cols_;
    }
    return *this;
  }

  MatrixX<T>& operator=(MatrixX<T>&& other) noexcept {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;
    }
    return *this;
  }

  MatrixX<T> operator+(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = data_[i] + scalar;
    }
    return result;
  }

  MatrixX<T> operator*(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
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

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = data_[i] + other(i);
    }

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
        T sum = 0;
        for (size_t k = 0; k < cols_; ++k) {
          sum += data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
        }
        result.data_[i * result.cols_ + j] = sum;
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

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = func(data_[i]);
    }
    return result;
  }

  template <typename U>
  MatrixX<U> elementwise(const MatrixX<T>& other, std::function<U(const T&, const T&)> func) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for element-wise division");
    }

    MatrixX<U> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = func(data_[i], other(i));
    }

    return result;
  }

  bool all(std::function<bool(const T&)> func) const {
    PARALLEL_ENABLED(1)
    for (size_t i = 0; i < numel(); ++i) {
      if (!func(data_[i])) {
        return false;
      }
    }
    return true;
  }

  bool all(const MatrixX<T>& other, std::function<bool(const T&, const T&)> func) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for element-wise comparison");
    }

    PARALLEL_ENABLED(1)
    for (size_t i = 0; i < numel(); ++i) {
      if (!func(data_[i], other(i))) {
        return false;
      }
    }
    return true;
  }

  bool any(std::function<bool(const T&)> func) const {
    return all([&func](const T& x) { return !func(x); });
  }
  bool any(const MatrixX<T>& other, std::function<bool(const T&, const T&)> func) {
    return all(other, [&func](const T& x, const T& y) { return !func(x, y); });
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
  MatrixX<T> sqrt() const {
    return elementwise<T>([](T x) { return std::sqrt(x); });
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

  std::pair<MatrixX<T>, MatrixX<T>> LU() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for LU decomposition");
    }

    L = MatrixX<T>(rows_, cols_);  // Initialize L as identity matrix
    U = *this;                     // Initialize U as a copy of the input matrix

    for (size_t k = 0; k < rows_; ++k) {
      // Find the pivot row
      size_t pivotRow = k;
      for (size_t i = k + 1; i < rows_; ++i) {
        if (std::abs(U(i, k)) > std::abs(U(pivotRow, k))) {
          pivotRow = i;
        }
      }

      // Swap rows in U matrix
      if (pivotRow != k) {
        std::swap(U.data[k], U.data[pivotRow]);
      }

      // Swap rows in L matrix
      std::swap(L.data[k], L.data[pivotRow]);

      // Perform elimination
      for (size_t i = k + 1; i < rows_; ++i) {
        T factor = U(i, k) / U(k, k);
        L(i, k) = factor;
        for (size_t j = k; j < cols_; ++j) {
          U(i, j) -= factor * U(k, j);
        }
      }
    }
    return std::make_pair(L, U);
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
      T temp = (*this)(i, k);
      (*this)(i, k) = (*this)(j, k);
      (*this)(j, k) = temp;
    }
  }

  // Swap two columns without using std::swap
  void swapCols(const size_t& i, const size_t& j) {
    if (i == j) {
      return;
    }

    PARALLEL_ENABLED(1)
    for (size_t k = 0; k < rows_; ++k) {
      T temp = (*this)(k, i);
      (*this)(k, i) = (*this)(k, j);
      (*this)(k, j) = temp;
    }
  }

  // gaussianElimination
  static MatrixX<T> GaussianElimination(const MatrixX<T>& matrix) {
    MatrixX<T> result = matrix;
    size_t n = matrix.rows();

    for (size_t i = 0; i < n; ++i) {
      // Find the pivot row
      size_t pivotRow = i;
      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(result(j, i)) > std::abs(result(pivotRow, i))) {
          pivotRow = j;
        }
      }

      // Swap rows
      if (pivotRow != i) {
        result.swapRows(i, pivotRow);
      }

      // Perform elimination
      for (size_t j = i + 1; j < n; ++j) {
        T factor = result(j, i) / result(i, i);
        for (size_t k = i; k < n; ++k) {
          result(j, k) -= factor * result(i, k);
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
    for (size_t i = 0; i < numel(); ++i) {
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

  /// Casting
  // cast to another type
  template <typename U>
  MatrixX<U> cast() const {
    MatrixX<U> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i, j) = static_cast<U>(data_[i]);
    }

    return result;
  }

  // cast to std::vector
  std::vector<T> toVector() const {
    std::vector<T> result(rows_ * cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = data_[i];
    }

    return result;
  }

  /// Resize
  // resize the matrix
  void resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows_ * cols_, 0.0);
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
    if (rows * cols != numel()) {
      throw std::invalid_argument("New shape must be compatible with the number of elements");
    }

    MatrixX<T> result(rows, cols);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < numel(); ++i) {
      result(i) = (*this)(i);
    }

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
  void print() { std::cout << *this; }
  friend std::ostream& operator<<(std::ostream& os, const MatrixX<T>& matrix) {
    os << std::setw(8) << std::fixed << std::setprecision(2) << "\n";
    for (size_t i = 0; i < matrix.rows(); ++i) {
      for (size_t j = 0; j < matrix.cols(); ++j) {
        os << matrix(i, j) << ", ";
      }
      os << std::endl;
    }
    return os;
  }

  /// Saving Data
  void save(std::ofstream& file) const {
    // Write the size of the outer vector
    size_t outerSize = data.size();
    file.write(reinterpret_cast<const char*>(&outerSize), sizeof(size_t));

    for (const auto& innerVec : data) {
      // Write the size of the inner vector
      size_t innerSize = innerVec.size();
      file.write(reinterpret_cast<const char*>(&innerSize), sizeof(size_t));

      // Write the elements of the inner vector
      file.write(reinterpret_cast<const char*>(innerVec.data()), sizeof(T) * innerSize);
    }
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
  size_t rows_;
  size_t cols_;
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
class randnX : public MatrixX<T> {
 public:
  randnX(const size_t& rows, const size_t& cols) : MatrixX<T>(rows, cols) { rand(); }
  randnX(const size_t& n) : MatrixX<T>(n, n) { rand(); }
};

template <typename T>
class randX : public MatrixX<T> {
 public:
  randX(const size_t& rows, const size_t& cols) : MatrixX<T>(rows, cols) { rand(); }
  randX(const size_t& n) : MatrixX<T>(n, n) { rand(); }
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
typedef randnX<double> randn;
typedef randnX<float> randnf;
typedef randX<double> rand;
typedef randX<float> randf;

}  // namespace matrix
#include "matrix/special_matrix.hpp"
