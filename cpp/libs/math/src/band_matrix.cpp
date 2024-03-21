#include "math/band_matrix.hpp"

namespace math {
BandMatrix::BandMatrix(int dim, int n_u, int n_l) { resize(dim, n_u, n_l); }

void BandMatrix::resize(int dim, int n_u, int n_l) {
  assert(dim > 0);
  assert(n_u >= 0);
  assert(n_l >= 0);
  U_.resize(n_u + 1);
  L_.resize(n_l + 1);
  for (auto& row : U_) row.resize(dim);
  for (auto& row : L_) row.resize(dim);
}

int BandMatrix::dim() const {
  if (U_.empty()) {
    return 0;
  } else {
    return static_cast<int>(U_[0].size());
  }
}

double& BandMatrix::operator()(int i, int j) {
  const int k = j - i;
  const int lower_limit = -num_lower();
  const int upper_limit = num_upper();
  const int dimension = dim();

  // Check if (i, j) is within the matrix bounds
  if (i < 0 || i >= dimension || j < 0 || j >= dimension || k < lower_limit || k > upper_limit) {
    throw std::out_of_range("Index out of bounds");
  }

  if (k >= 0) {
    return U_[k][i];
  } else {
    return L_[-k][i];
  }
}

double BandMatrix::operator()(int i, int j) const {
  const int k = j - i;
  const int lower_limit = -num_lower();
  const int upper_limit = num_upper();
  const int dimension = dim();

  // Check if (i, j) is within the matrix bounds
  if (i < 0 || i >= dimension || j < 0 || j >= dimension || k < lower_limit || k > upper_limit) {
    throw std::out_of_range("Index out of bounds");
  }

  if (k >= 0) {
    return U_[k][i];
  } else {
    return L_[-k][i];
  }
}

double BandMatrix::saved_diag(int i) const {
  assert(i >= 0 && i < dim());
  return L_[0][i];  // Return the diagonal element at position i
}

double& BandMatrix::saved_diag(int i) {
  assert(i >= 0 && i < dim());
  return L_[0][i];
}

void BandMatrix::lu_decompose() {
  for (int i = 0; i < dim(); ++i) {
    assert((*this)(i, i) != 0.0);
    double inv_diag = 1.0 / (*this)(i, i);
    saved_diag(i) = inv_diag;
    int j_min = std::max(0, i - num_lower());
    int j_max = std::min(dim() - 1, i + num_upper());

    // Normalize the current row
    for (int j = j_min; j <= j_max; ++j) (*this)(i, j) *= inv_diag;

    (*this)(i, i) = 1.0;  // Set diagonal to 1
  }

  for (int k = 0; k < dim(); ++k) {
    int i_max = std::min(dim() - 1, k + num_lower());

    // Eliminate lower-band elements
    for (int i = k + 1; i <= i_max; ++i) {
      assert((*this)(k, k) != 0.0);
      double factor = -(*this)(i, k) / (*this)(k, k);
      (*this)(i, k) = -factor;
      int j_max = std::min(dim() - 1, k + num_upper());

      // Update the elements in the current row
      for (int j = k + 1; j <= j_max; ++j) (*this)(i, j) += factor * (*this)(k, j);
    }
  }
}

std::vector<double> BandMatrix::l_solve(const std::vector<double>& b) const {
  assert(dim() == static_cast<int>(b.size()));
  std::vector<double> x(dim());

  for (int i = 0; i < dim(); ++i) {
    double sum = 0;
    int j_start = std::max(0, i - num_lower());

    // Compute the sum of products without accessing the elements directly
    for (int j = j_start; j < i; ++j) sum += (*this)(i, j) * x[j];

    // Apply the saved diagonal and subtract the sum to compute x[i]
    x[i] = b[i] * saved_diag(i) - sum;
  }

  return x;
}

std::vector<double> BandMatrix::r_solve(const std::vector<double>& b) const {
  assert(dim() == static_cast<int>(b.size()));
  std::vector<double> x(dim());

  for (int i = dim() - 1; i >= 0; --i) {
    double sum = 0;
    int j_stop = std::min(dim() - 1, i + num_upper());

    // Compute the sum of products without accessing the elements directly
    for (int j = i + 1; j <= j_stop; ++j) sum += (*this)(i, j) * x[j];

    // Apply the saved diagonal and compute x[i]
    x[i] = (b[i] - sum) / (*this)(i, i);
  }

  return x;
}

std::vector<double> BandMatrix::lu_solve(const std::vector<double>& b, bool is_lu_decomposed) {
  assert(dim() == static_cast<int>(b.size()));

  // Perform LU decomposition if needed
  if (!is_lu_decomposed) lu_decompose();

  // Solve Ly = b
  std::vector<double> y = l_solve(b);

  // Solve Ux = y
  return r_solve(y);
}

}  // namespace math
