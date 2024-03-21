#include <iostream>
#include "matrix/matrix.hpp"

struct TCB {
  int state;
  std::string name;
};

int main() {
  const size_t matrixSize = 3;
  const double lowerBound = -10.0;
  const double upperBound = 10.0;

  matrix::Matrix A(matrixSize, matrixSize);
  A.setIdentity();
  A(1, 1) = -2;

  matrix::Matrix B(matrixSize, matrixSize);
  B.setIdentity();
  B(1, 1) = -3;

  TCB tcb;
  tcb.state = 0;
  tcb.name = "blah";

  std::cout << "A: " << A << std::endl;
  std::cout << "B: " << B << std::endl;
  std::cout << "A^T: " << A.transpose() << std::endl;
  std::cout << "A^-1: " << A.inverse() << std::endl;
  std::cout << "A.abs(): " << A.abs() << std::endl;
  std::cout << "A.maxCoeff(): " << A.maxCoeff() << std::endl;
  std::cout << "A.minCoeff(): " << A.minCoeff() << std::endl;
  std::cout << "A.sum(): " << A.sum() << std::endl;
  std::cout << "A.mean(): " << A.mean() << std::endl;
  std::cout << "A.prod(): " << A.prod() << std::endl;
  std::cout << "A.trace(): " << A.trace() << std::endl;
  std::cout << "A.diagonal(): " << A.diagonal() << std::endl;
  std::cout << "A.determinant(): " << A.determinant() << std::endl;

  std::cout << "A(4): " << A(4) << std::endl;
  std::cout << "A: " << A << std::endl;

  std::cout << "A + 1: " << A + 1 << std::endl;
  std::cout << "1 + A: " << 1 + A << std::endl;

  std::cout << "A - 1: " << A - 1 << std::endl;
  std::cout << "1 - A: " << 1 - A << std::endl;

  std::cout << "2 * A: " << 2 * A << std::endl;
  std::cout << "A * 2: " << A * 2 << std::endl;

  std::cout << "2 / A: " << 2 / A << std::endl;
  std::cout << "A / 2: " << A / 2 << std::endl;

  std::cout << "A + B: " << A + B << std::endl;
  std::cout << "A - B: " << A - B << std::endl;
  std::cout << "A * B: " << A * B << std::endl;
  std::cout << "A / B: " << A / B << std::endl;

  return 0;
}
