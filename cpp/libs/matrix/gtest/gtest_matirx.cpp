#include <iostream>

#include <gtest/gtest.h>

#include "matrix/matrix.hpp"
#include "time/time.hpp"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(Matrix, Constructor) {
  matrix::Matrix A(3, 3);
  EXPECT_EQ(A.rows(), 3);
  EXPECT_EQ(A.cols(), 3);

  matrix::Matrix B(3, 3, 1);
  EXPECT_EQ(B.rows(), 3);
  EXPECT_EQ(B.cols(), 3);
  for (size_t i = 0; i < B.rows(); i++) {
    for (size_t j = 0; j < B.cols(); j++) {
      EXPECT_EQ(B(i, j), 1);
    }
  }
}

TEST(Matrix, Eye) {
  matrix::eye A(3, 3);
  EXPECT_EQ(A.rows(), 3);
  EXPECT_EQ(A.cols(), 3);
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      if (i == j) {
        EXPECT_EQ(A(i, j), 1);
      } else {
        EXPECT_EQ(A(i, j), 0);
      }
    }
  }
}

TEST(Matrix, Transpose) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix At = A.transpose();
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_EQ(A(i, j), At(j, i));
    }
  }
}

TEST(Matrix, Inverse) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {-1, 8, 9}};
  matrix::Matrix Ainv = A.inv();
  matrix::Matrix I = A * Ainv;
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      if (i == j) {
        EXPECT_NEAR(I(i, j), 1, 1e-4);
      } else {
        EXPECT_NEAR(I(i, j), 0, 1e-4);
      }
    }
  }
}

TEST(Matrix, Add) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix B{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
  matrix::Matrix C = A + B;
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_EQ(C(i, j), A(i, j) + B(i, j));
    }
  }
}

TEST(Matrix, Subtract) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix B{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
  matrix::Matrix C = A - B;
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_EQ(C(i, j), A(i, j) - B(i, j));
    }
  }
}

TEST(Matrix, Multiply) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix B{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
  matrix::Matrix C = A * B;
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      double sum = 0;
      for (size_t k = 0; k < A.cols(); k++) {
        sum += A(i, k) * B(k, j);
      }
      EXPECT_EQ(C(i, j), sum);
    }
  }
}

TEST(Matrix, Abs) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix B = A.abs();
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_EQ(B(i, j), std::abs(A(i, j)));
    }
  }
}

TEST(Matrix, Max) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.max(), 9);
}

TEST(Matrix, Min) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.min(), -3);
}

TEST(Matrix, Sum) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.sum(), 33);
}

TEST(Matrix, Mean) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.mean(), 3.6666666666666665);
}

TEST(Matrix, Prod) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.prod(), -362880);
}

TEST(Matrix, Trace) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.trace(), 13);
}

TEST(Matrix, Diag) {
  matrix::Matrix A{{-1, -2, -3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix diag = A.diag();
  for (size_t i = 0; i < A.rows(); i++) {
    EXPECT_EQ(diag(i), A(i, i));
  }
}

TEST(Matrix, Det) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  EXPECT_EQ(A.det(), 0);
}

TEST(Matrix, Col) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix col = A.col(0);
  for (size_t i = 0; i < A.rows(); i++) {
    EXPECT_EQ(col(i), A(i, 0));
  }
}

TEST(Matrix, GaussianElimination) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix::Matrix B = {{7, 8, 9}, {0, 0.86, 1.71}, {0, 0, 0}};
  matrix::Matrix G = matrix::Matrix::GaussianElimination(A);
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_NEAR(G(i, j), B(i, j), 1e-2);
    }
  }
}

TEST(Matrix, QR) {
  matrix::Matrix A{{1, 2, 3}, {4, 5, 6}, {1, 8, 9}};
  std::pair<matrix::Matrix, matrix::Matrix> qr = A.qr();
  matrix::Matrix Q = qr.first;
  matrix::Matrix R = qr.second;
  matrix::Matrix I = Q.transpose() * Q;
  for (size_t i = 0; i < I.rows(); i++) {
    for (size_t j = 0; j < I.cols(); j++) {
      if (i == j) {
        EXPECT_NEAR(I(i, j), 1, 1e-6);
      } else {
        EXPECT_NEAR(I(i, j), 0, 1e-6);
      }
    }
  }
  matrix::Matrix C = Q * R;
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      EXPECT_NEAR(A(i, j), C(i, j), 1e-6);
    }
  }
}

TEST(Matrix, Strassen) {
  size_t N = 100;
  matrix::rand X(N, N);
  matrix::rand Y(N, N);
  matrix::Matrix Z = X.strassen(Y);
  matrix::Matrix Z2 = X * Y;
  for (size_t i = 0; i < Z.rows(); i++) {
    for (size_t j = 0; j < Z.cols(); j++) {
      EXPECT_NEAR(Z(i, j), Z2(i, j), 1e-6);
    }
  }
}
