#include "matrix/special_matrix.hpp"

#include "matrix/qr.hpp"
#include "time/time.hpp"
#include <string>

#include <cstdio>  // Include the header file for printf

template <typename T>
void qr_alg(
    const std::string& method,
    std::function<std::pair<matrix::MatrixX<T>, matrix::MatrixX<T>>(const matrix::MatrixX<T>&)>
        func) {
  auto A = matrix::Hilbert<T>(100);

  printf("\n%s Method: %zdx%zd\n", method.c_str(), A.rows(), A.cols());
  tic();
  auto QR = func(A);
  double elapsed = mtoc();  // Calculate elapsed time in microseconds
  printf("Time=%f msec Error: %4.4e \n", elapsed, (A - QR.first * QR.second).norm());
};

int main() {
  qr_alg<double>("Classic", matrix::qr::Classic<double>);
  qr_alg<double>("ClassicGS", matrix::qr::ClassicGS<double>);
  qr_alg<double>("ModifiedGS", matrix::qr::ModifiedGS<double>);
  return 0;
}