#include <iostream>
#include "matrix/matrix.hpp"
#include "time/time.hpp"
#include "Eigen/Dense"
int main() {
  std::cout << math::linspace(0.0, 1.0, 2);

  const size_t matrixSize = 3;
  const double lowerBound = -10.0;
  const double upperBound = 10.0;

  // Construct with special matrix with random values
  {
    matrix::Matrix A(matrixSize, matrixSize);
    matrix::eye A1(matrixSize, matrixSize);
    matrix::rand A2(matrixSize, matrixSize, lowerBound, upperBound);
    matrix::ones A3(matrixSize, matrixSize);
    matrix::zeros A4(matrixSize, matrixSize);
  }

  {
    matrix::rand A(matrixSize, matrixSize);
    {
      std::cout << "A: " << A << std::endl;
      std::cout << "A(4): " << A(4) << std::endl;
    }

    {
      std::cout << "A + 1: " << A + 1 << std::endl;
      std::cout << "1 + A: " << 1 + A << std::endl;

      std::cout << "A - 1: " << A - 1 << std::endl;
      std::cout << "1 - A: " << 1 - A << std::endl;

      std::cout << "2 * A: " << 2 * A << std::endl;
      std::cout << "A * 2: " << A * 2 << std::endl;

      std::cout << "2 / A: " << 2 / A << std::endl;
      std::cout << "A / 2: " << A / 2 << std::endl;
    }

    {
      std::cout << "A.abs(): " << A.abs() << std::endl;
      std::cout << "A.max(): " << A.max() << std::endl;
      std::cout << "A.min(): " << A.min() << std::endl;
      std::cout << "A.sum(): " << A.sum() << std::endl;
      std::cout << "A.mean(): " << A.mean() << std::endl;
      std::cout << "A.prod(): " << A.prod() << std::endl;
      std::cout << "A.trace(): " << A.trace() << std::endl;
      std::cout << "A.diag(): " << A.diag() << std::endl;
      std::cout << "A.det(): " << A.det() << std::endl;
    }

    {
      std::cout << "A^T: " << A.transpose() << std::endl;
      std::cout << "A^-1: " << A.inv() << std::endl;
    }

    {
      matrix::rand B(matrixSize, matrixSize);
      {
        std::cout << "A.dot(B): " << A.dot(B) << std::endl;
        std::cout << "A + B: " << A + B << std::endl;
        std::cout << "A - B: " << A - B << std::endl;
        std::cout << "A * B: " << A * B << std::endl;
        std::cout << "A / B: " << A / B << std::endl;
      }
    }
  }

  {
    matrix::Matrix C{{1, 1, 3}, {4, 1, 6}, {7, 8, 9}};
    {
      std::cout << "C.col(0)" << C.col(0);
      std::cout << "GaussianElimination" << matrix::Matrix::GaussianElimination(C) << std::endl;
    }

    {
      std::pair<matrix::Matrix, matrix::Matrix> qr = C.qr();
      std::cout << "Q: " << qr.first << std::endl;
      std::cout << "R: " << qr.second << std::endl;

      std::cout << "Q * Qt: " << qr.first.transpose() * qr.first << std::endl;
      std::cout << "C: " << qr.first * qr.second << std::endl;
    }
  }

  // matrix::Matrix E{{1, 2}, {1, 1}};
  // std::cout << "E" << E << std::endl;

  // std::cout << "E.eig(): \n" << E.eig() << std::endl;

  // matrix::rand F(100, 100);
  // auto DV = F.eigen();
  // std::cout << "D: " << std::get<0>(DV) << std::endl;
  // std::cout << "V: " << std::get<1>(DV) << std::endl;

  // std::cout << "E" << E << std::endl;
  // std::tuple<matrix::Matrix, matrix::Matrix, matrix::Matrix> SVD = (E.transpose() * E).svd();

  // std::cout << "U: " << std::get<0>(SVD) << std::endl;
  // std::cout << "S: " << std::get<1>(SVD) << std::endl;
  // std::cout << "V: " << std::get<2>(SVD) << std::endl;

  // std::cout << "U * S * V.transpose(): "
  //           << std::get<0>(SVD) * std::get<1>(SVD) * std::get<2>(SVD).transpose() << std::endl;

  // {
  //   size_t N = 500;
  //   matrix::rand X(N, N);
  //   matrix::rand Y(N, N);
  //   // X(1, 1) = 2;
  //   // Y(1, 1) = -1;

  //   tic();
  //   auto Z = X * Y;
  //   toc();

  //   std::cout << "Z=X*Y: " << Z.norm() << " \n Elapse time:" << toc() << std::endl;
  // }

  // {
  //   size_t N = 500;
  //   matrix::rand X(N, N);
  //   X.save("X.txt");
  //   matrix::Matrix Y = matrix::Matrix::Load("X.txt");
  //   std::cout << "error: " << (X - Y).norm() << std::endl;
  // }

  {
    size_t N = 200;
    matrix::rand X(N, N);
    matrix::rand Y(N, N);

    {
      tic();
      matrix::Matrix Z(N, N);
      Z = X * Y;
      auto elapse = mtoc();
      std::cout << "Matrix -> Z=X*Y: " << Z.norm() << " \n Elapse time:" << elapse << std::endl;
    }
    {
      Eigen::MatrixXd X1(N, N);
      Eigen::MatrixXd Y1(N, N);
      X1 = Eigen::Map<Eigen::MatrixXd>(X.data(), N, N);
      Y1 = Eigen::Map<Eigen::MatrixXd>(Y.data(), N, N);
      std::cout << "start\n";
      tic();
      Eigen::MatrixXd Z(N, N);
      Z = X1 * Y1;
      auto elapse = mtoc();
      std::cout << "Matrix -> Z=X*Y: " << Z.norm() << " \n Elapse time:" << elapse << std::endl;
    }
  }

  // tic();
  // auto Z2 = X.strassen(Y);
  // toc();
  // std::cout << "X*Y: " << Z2.norm() << " \n Elapse time:" << toc() << std::endl;

  return 0;
}
