#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <functional>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <iomanip>  // for std::setw

#define PARALLEL_ENABLED 1  // Set to 1 to enable parallel computing, 0 to disable

#if PARALLEL_ENABLED > 0
#define PARALLEL_FOR_COLLAPSE(N) \
    _Pragma(STRINGIFY(omp parallel for collapse(N)))
#define STRINGIFY(x) #x
#else
#define PARALLEL_FOR_COLLAPSE(N)
#endif

namespace matrix {

template <typename T = double>
class MatrixX {
 public:
  // Constructor
  MatrixX() : rows_(0), cols_(0) {}
  MatrixX(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    data_.resize(rows_, std::vector<T>(cols_, 0.0));
  };
  MatrixX(size_t rows, size_t cols, T value) : rows_(rows), cols_(cols) {
    data_.resize(rows_, std::vector<T>(cols_, value));
  }

  size_t rows() const { return rows_; }

  size_t cols() const { return cols_; }

  // Random initialization
  void setRandom(T lowerBound = -1.0, T upperBound = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lowerBound, upperBound);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        data_[i][j] = dis(gen);
      }
    }
  }

  // Method to generate an zero matrix
  void setZero() {
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        data_[i][j] = 0.0;
      }
    }
  }

  // Method to generate an identity matrix
  void setIdentity() {
    setZero();
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < std::min(rows_, cols_); ++i) {
      data_[i][i] = 1.0;
    }
  }

  void setOne() {
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        data_[i][j] = 1.0;
      }
    }
  }

  // Apply element-wise function
  MatrixX<T> bitfunc(std::function<T(T)> func) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = func(data_[i][j]);
      }
    }

    return result;
  }

  // Overloaded operator for element-wise multiplication
  MatrixX<T> elementwiseProd(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument(
          "MatrixX dimensions do not match for element-wise multiplication");
    }

    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = data_[i][j] * other(i, j);
      }
    }

    return result;
  }

  // Overloaded operator for element-wise division
  MatrixX<T> elementwiseDiv(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for element-wise division");
    }

    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (other(i, j) == 0) {
          throw std::invalid_argument("Division by zero");
        }
        result(i, j) = data_[i][j] / other(i, j);
      }
    }

    return result;
  }

  // Transpose
  MatrixX<T> transpose() const {
    MatrixX<T> result(cols_, rows_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < cols_; ++i) {
      for (size_t j = 0; j < rows_; ++j) {
        result(i, j) = data_[j][i];
      }
    }

    return result;
  }

  // Diagonal
  std::vector<T> diagonal() const {
    std::vector<T> result(std::min(rows_, cols_));

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < std::min(rows_, cols_); ++i) {
      result[i] = data_[i][i];
    }

    return result;
  }

  // MatrixX inverse using Gaussian elimination (without pivoting)
  MatrixX<T> inverse() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for inverse calculation");
    }

    size_t n = rows_;

    // Augmenting the matrix with identity matrix of the same size
    MatrixX<T> augmented(n, 2 * n);
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        augmented(i, j) = data_[i][j];
        augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
      }
    }

    // Forward elimination
    PARALLEL_FOR_COLLAPSE(3)
    for (size_t i = 0; i < n; ++i) {
      // Divide the row by the diagonal element to make it 1
      T pivot = augmented(i, i);
      if (pivot == 0) {
        throw std::runtime_error("MatrixX is singular, inverse does not exist");
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

    // Extracting the inverse from the augmented matrix
    MatrixX<T> result(n, n);
    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result(i, j) = augmented(i, j + n);
      }
    }

    return result;
  }

  T determinant() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("MatrixX must be square for determinant calculation");
    }

    size_t n = rows_;
    MatrixX<T> temp(*this);  // Create a copy of the input matrix to avoid modifying it

    // Forward elimination
    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < n; ++i) {
      T pivot = temp(i, i);
      if (pivot == 0) {
        return 0;
      }

      // Divide the row by the diagonal element to make it 1
      for (size_t j = 0; j < n; ++j) {
        temp(i, j) /= pivot;
      }

      // Eliminate non-zero elements below the diagonal
      for (size_t k = i + 1; k < n; ++k) {
        T factor = temp(k, i);
        for (size_t j = 0; j < n; ++j) {
          temp(k, j) -= factor * temp(i, j);
        }
      }
    }

    // Calculate determinant
    T det = 1.0;
    for (size_t i = 0; i < n; ++i) {
      det *= temp(i, i);
    }

    return det;
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

  T& operator()(const size_t& i, const size_t& j) { return data_[i][j]; }

  const T& operator()(const size_t& i, const size_t& j) const { return data_[i][j]; }

  friend std::ostream& operator<<(std::ostream& os, const MatrixX& matrix) {
    os << "[\n";
    for (size_t i = 0; i < matrix.rows_; ++i) {
      for (size_t j = 0; j < matrix.cols_; ++j) {
        os << std::setw(8) << std::fixed << std::setprecision(2) << matrix.data_[i][j] << " ";
      }
      os << std::endl;
    }
    os << "]";
    return os;
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

  MatrixX<T> operator+(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = data_[i][j] + scalar;
      }
    }
    return result;
  }

  MatrixX<T> operator*(const T& scalar) const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = data_[i][j] * scalar;
      }
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

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = data_[i][j] + other(i, j);
      }
    }

    return result;
  }

  // MatrixX multiplication
  MatrixX<T> operator*(const MatrixX<T>& other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument("MatrixX dimensions are not compatible for dot product");
    }

    MatrixX<T> result(rows_, other.cols_);

    PARALLEL_FOR_COLLAPSE(3)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < other.cols_; ++j) {
        for (size_t k = 0; k < cols_; ++k) {
          result(i, j) += data_[i][k] * other(k, j);
        }
      }
    }

    return result;
  }

  MatrixX<T> operator-(const MatrixX<T>& other) const { return (*this) + (-1 * other); };
  MatrixX<T> operator/(const MatrixX<T>& other) { return *this * other.inverse(); };

  MatrixX<T> operator+=(const MatrixX<T>& other) { return *this = *this + other; };
  MatrixX<T> operator-=(const MatrixX<T>& other) { return *this = *this - other; };
  MatrixX<T> operator*=(const MatrixX<T>& other) { return *this = *this * other; };

  /// element-wise division
  friend MatrixX<T> operator/(const T& scalar, const MatrixX<T>& matrix) {
    return scalar * matrix.inverse();
  }

  // Overloaded operator* for matrix-vector multiplication
  std::vector<T> operator*(const std::vector<T>& vec) const {
    if (this->cols() != vec.size()) {
      throw std::invalid_argument(
          "Vector size does not match matrix dimensions for multiplication");
    }

    std::vector<T> result(this->rows(), 0.0);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < this->rows(); ++i) {
      for (size_t j = 0; j < this->cols(); ++j) {
        result[i] += (*this)(i, j) * vec[j];
      }
    }

    return result;
  }

  // Call element [] operator
  T& operator[](const size_t& index) {
    return data_[index % rows_][index / rows_];  // Adjusted for column-major order
  }

  const T& operator[](const size_t& index) const {
    return this->data_[index % rows_][index / rows_];
  }

  T& operator()(const size_t& index) {
    return data_[index % rows_][index / rows_];  // Adjusted for column-major order
  }

  const T& operator()(const size_t& index) const {
    return this->data_[index % rows_][index / rows_];
  }

  bool operator==(const MatrixX<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      return false;
    }
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (data_[i][j] != other(i, j)) {
          return false;
        }
      }
    }
    return true;
  }

  // Boolean greater than operator
  MatrixX<T> operator>(const T& scalar) const {
    return bitfunc([scalar](T x) { return x > scalar; });
  }

  // Boolean less than operator
  MatrixX<T> operator<(const T& scalar) const {
    return bitfunc([scalar](T x) { return x < scalar; });
  }

  // Boolean greater than or equal to operator
  MatrixX<T> operator>=(const T& scalar) const {
    return bitfunc([scalar](T x) { return x >= scalar; });
  }

  // Boolean less than or equal to operator
  MatrixX<T> operator<=(const T& scalar) const {
    return bitfunc([scalar](T x) { return x <= scalar; });
  }

  // Boolean equal to operator
  MatrixX<T> operator==(const T& scalar) const {
    return bitfunc([scalar](T x) { return x == scalar; });
  }

  // Boolean not equal to operator
  MatrixX<T> operator!=(const T& scalar) const {
    return bitfunc([scalar](T x) { return x != scalar; });
  }

  // Boolean greater than operator
  MatrixX<T> operator>(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x > other.data_[0][0]; });
  }

  // Boolean less than operator

  MatrixX<T> operator<(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x < other.data_[0][0]; });
  }

  // Boolean greater than or equal to operator

  MatrixX<T> operator>=(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x >= other.data_[0][0]; });
  }

  // Boolean less than or equal to operator

  MatrixX<T> operator<=(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x <= other.data_[0][0]; });
  }

  // Boolean equal to operator

  MatrixX<T> operator==(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x == other.data_[0][0]; });
  }

  // Boolean not equal to operator

  MatrixX<T> operator!=(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for comparison");
    }

    return bitfunc([&other](const T& x) { return x != other.data_[0][0]; });
  }

  T norm() const {
    T result = 0.0;

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result += data_[i][j] * data_[i][j];
      }
    }

    return std::sqrt(result);
  }

  MatrixX<T> abs() const {
    return bitfunc([](T x) { return std::abs(x); });
  }

  MatrixX<T> sin() const {
    return bitfunc([](T x) { return std::sin(x); });
  }

  MatrixX<T> cos() const {
    return bitfunc([](T x) { return std::cos(x); });
  }

  MatrixX<T> tan() const {
    return bitfunc([](T x) { return std::tan(x); });
  }

  MatrixX<T> asin() const {
    return bitfunc([](T x) { return std::asin(x); });
  }

  MatrixX<T> acos() const {
    return bitfunc([](T x) { return std::acos(x); });
  }

  MatrixX<T> atan() const {
    return bitfunc([](T x) { return std::atan(x); });
  }

  MatrixX<T> normalized() const {
    T norm_ = norm();
    if (norm_ == 0) {
      throw std::invalid_argument("Division by zero");
    }
    return *this / norm_;
  }

  T dot(const MatrixX<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("MatrixX dimensions do not match for dot product");
    }

    T result = 0.0;

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result += data_[i][j] * other(i, j);
      }
    }

    return result;
  }

  T maxCoeff() const {
    T result = data_[0][0];

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (data_[i][j] > result) {
          result = data_[i][j];
        }
      }
    }
    return result;
  }

  T minCoeff() const {
    T result = data_[0][0];

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (data_[i][j] < result) {
          result = data_[i][j];
        }
      }
    }
    return result;
  }

  T sum() const {
    T result = 0.0;

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result += data_[i][j];
      }
    }
    return result;
  }

  T mean() const { return sum() / (rows_ * cols_); }

  T prod() const {
    T result = 1.0;

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result *= data_[i][j];
      }
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
      result += data_[i][i];
    }
    return result;
  }

  MatrixX<T> tanh() const {
    return bitfunc([](T x) { return std::tanh(x); });
  }

  // sigmoid function
  MatrixX<T> sigmoid() const {
    return bitfunc([](T x) { return 1.0 / (1.0 + std::exp(-x)); });
  }

  MatrixX<T> relu() const {
    return bitfunc([](T x) { return (x > 0) ? x : 0; });
  }

  //  square root function
  MatrixX<T> sqrt() const {
    return bitfunc([](T x) { return std::sqrt(x); });
  }

  //  exponential function
  MatrixX<T> exp() const {
    return bitfunc([](T x) { return std::exp(x); });
  }

  //  log function
  MatrixX<T> log() const {
    return bitfunc([](T x) { return std::log(x); });
  }

  // square function
  MatrixX<T> square() const {
    return bitfunc([](T x) { return x * x; });
  }

  MatrixX<T> softmax() const {
    MatrixX<T> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(1)
    for (size_t i = 0; i < rows_; ++i) {
      T sum = 0.0;
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = std::exp(data_[i][j]);
        sum += result(i, j);
      }
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) /= sum;
      }
    }
    return result;
  }

  // cast to another type
  template <typename U>
  MatrixX<U> cast() const {
    MatrixX<U> result(rows_, cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = static_cast<U>(data_[i][j]);
      }
    }

    return result;
  }

  // cast to std::vector
  std::vector<T> toVector() const {
    std::vector<T> result(rows_ * cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result[i * cols_ + j] = data_[i][j];
      }
    }

    return result;
  }

  // resize the matrix
  void resize(size_t rows, size_t cols) {
    data_.resize(rows, std::vector<T>(cols, 0.0));
    rows_ = rows;
    cols_ = cols;
  }

  // get data
  std::vector<std::vector<T>> vec() const { return data_; }

  // get data array
  std::vector<T> array() const {
    std::vector<T> result(rows_ * cols_);

    PARALLEL_FOR_COLLAPSE(2)
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result[i * cols_ + j] = data_[i][j];
      }
    }

    return result;
  }

  // get raw data
  T* data() { return data_[0].data(); }

 protected:
  std::vector<std::vector<T>> data_;
  size_t rows_;
  size_t cols_;
};

typedef MatrixX<double> Matrix;
typedef MatrixX<float> Matrixf;

}  // namespace matrix

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
  os << "[";
  for (size_t i = 0; i < vector.size(); ++i) {
    os << std::setw(8) << std::fixed << std::setprecision(2) << vector[i] << " ";
  }
  os << "]";
  return os;
}
