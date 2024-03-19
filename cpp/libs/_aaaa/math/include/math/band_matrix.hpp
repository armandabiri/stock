#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>

namespace math {
class BandMatrix {
 private:
  std::vector<std::vector<double>> U_;  // upper band
  std::vector<std::vector<double>> L_;  // lower band

 public:
  // Constructors and Destructor
  BandMatrix() = default;
  BandMatrix(int dim, int n_u, int n_l);
  ~BandMatrix() = default;

  // Methods
  void resize(int dim, int n_u, int n_l);  // Initialize with dim, n_u, n_l
  int dim() const;                         // Matrix dimension
  int num_upper() const { return static_cast<int>(U_.size()) - 1; }
  int num_lower() const { return static_cast<int>(L_.size()) - 1; }

  // Access Operators
  double& operator()(int i, int j);       // Write
  double operator()(int i, int j) const;  // Read

  // Additional Diagonal
  double& saved_diag(int i);
  double saved_diag(int i) const;

  // LU Decomposition and Solvers
  void lu_decompose();
  std::vector<double> r_solve(const std::vector<double>& b) const;
  std::vector<double> l_solve(const std::vector<double>& b) const;
  std::vector<double> lu_solve(const std::vector<double>& b, bool is_lu_decomposed = false);
};
}  // namespace math
