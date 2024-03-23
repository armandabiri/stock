#pragma once

#include "matrix/matrix.hpp"

namespace matrix {
// hilbert matrix
template <typename T>
MatrixX<T> Hilbert(const size_t& n) {
  MatrixX<T> H(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      H(i, j) = 1.0 / (i + j + 1);
    }
  }
  return H;
}

// vandermonde matrix
template <typename T>
MatrixX<T> Vandermonde(const VectorX<T>& x) {
  const size_t n = x.numel();
  MatrixX<T> V(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      V(i, j) = std::pow(x(i), j);
    }
  }
  return V;
}

// toeplitz matrix
template <typename T>
MatrixX<T> Toeplitz(const VectorX<T>& c) {
  const size_t n = c.numel();
  MatrixX<T> T(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T(i, j) = c(std::abs(i - j));
    }
  }
  return T;
}

// circulant matrix
template <typename T>
MatrixX<T> Circulant(const VectorX<T>& c) {
  const size_t n = c.numel();
  MatrixX<T> C(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C(i, j) = c(std::abs(i - j));
    }
  }
  return C;
}

// cauchy matrix
template <typename T>
MatrixX<T> Cauchy(const VectorX<T>& x, const VectorX<T>& y) {
  const size_t m = x.numel();
  const size_t n = y.numel();
  MatrixX<T> C(m, n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C(i, j) = 1.0 / (x(i) - y(j));
    }
  }
  return C;
}

// lehmer matrix
template <typename T>
MatrixX<T> Lehmer(const size_t& n) {
  MatrixX<T> L(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      L(i, j) = std::min(i, j) + 1;
    }
  }
  return L;
}

// pascal matrix
template <typename T>
MatrixX<T> Pascal(const size_t& n) {
  MatrixX<T> P(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      P(i, j) = std::tgamma(i + j + 2) / (std::tgamma(i + 1) * std::tgamma(j + 1));
    }
  }
  return P;
}

}  // namespace matrix