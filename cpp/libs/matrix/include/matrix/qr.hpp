#pragma once

#include "matrix/matrix.hpp"

namespace matrix {
namespace qr {

// Classical Gram-Schmidt orthogonalization
template <typename T>
inline std::pair<MatrixX<T>, MatrixX<T>> Classic(const MatrixX<T>& A) {
  const size_t m = A.rows();
  const size_t n = A.cols();
  auto R = MatrixX<T>(n, n);

  R.zeros();
  auto Q = A;

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

template <typename T>
inline std::pair<MatrixX<T>, MatrixX<T>> ModifiedGS(const MatrixX<T>& A) {
  const size_t m = A.rows();
  const size_t n = A.cols();
  MatrixX<T> R(n, n);

  R.zeros();
  auto Q = A;

  PARALLEL_FOR_COLLAPSE(2)
  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < k; ++i) {
      R(i, k) = Q.col(i).dot(Q.col(k));
      Q.setCol(k, Q.col(i) - Q.col(i) * R(i, k));  // Subtract projection
    }
    R(k, k) = Q.col(k).norm();
    Q.setCol(k, Q.col(k) / R(k, k));
  }

  return std::make_pair(Q, R);
}

template <typename T>
inline std::pair<MatrixX<T>, MatrixX<T>> ClassicGS(const MatrixX<T>& A) {
  const size_t m = A.rows();
  const size_t n = A.cols();
  MatrixX<T> Q = A;
  MatrixX<T> R(n, n);

  R.zeros();

  for (size_t k = 0; k < n; ++k) {
    // Compute R(1:k-1, k)
    for (size_t i = 0; i < k; ++i) {
      R(i, k) = Q.col(i).dot(Q.col(k));
    }
    // Update Q(:,k) and compute R(k,k)
    for (size_t i = 0; i < k; ++i) {
      Q.setCol(k, Q.col(i) - Q.col(i) * R(i, k));  // Subtract projection
    }
    R(k, k) = Q.col(k).norm();  // Compute norm
    Q.setCol(k, Q.col(k) / R(k, k));
  }

  return std::make_pair(Q, R);
}

}  // namespace qr
}  // namespace matrix