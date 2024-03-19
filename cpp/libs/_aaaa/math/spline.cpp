#include "math/spline.hpp"

namespace math {
void Spline::set_boundary(const BoundaryType& left, const double& left_value,
                          const BoundaryType& right, const double& right_value) {
  assert(X_.empty());  // set_points() must not have happened yet
  bc_left_ = left;
  bc_right_ = right;
  left_value_ = left_value;
  right_value_ = right_value;
}

void Spline::set_coeffs_from_b() {
  assert(X_.size() == Y_.size());
  assert(X_.size() == bs_.size());
  assert(static_cast<int>(X_.size()) > 2);
  const size_t n = bs_.size();

  cs_.reserve(n);
  ds_.reserve(n);

  const double one_third = 1.0 / 3.0;
  const double two_third = 2.0 / 3.0;

  for (size_t i = 0; i < n - 1; i++) {
    const double h = X_[i + 1] - X_[i];
    const double diff_y = Y_[i + 1] - Y_[i];
    const double diff_b = bs_[i + 1] - bs_[i];

    cs_.push_back((3.0 * diff_y / h - (2.0 * bs_[i] + bs_[i + 1])) / h);
    ds_.push_back((diff_b / (3.0 * h) - two_third * cs_[i]) / h);
  }

  c0_ = (bc_left_ == BoundaryType::FirstDerivative) ? 0.0 : cs_.front();
}

void Spline::set_points(const std::vector<double>& x, const std::vector<double>& y,
                        const SplineType& type) {
  assert(x.size() == y.size());
  assert(x.size() > 2);
  spline_type_ = type;
  monotonic_ = false;
  X_ = x;
  Y_ = y;
  int n = static_cast<int>(x.size());

  for (int i = 0; i < n - 1; i++) {
    assert(X_[i] < X_[i + 1]);
  }

  if (type == SplineType::Linear) {
    ds_.resize(n);
    cs_.resize(n);
    bs_.resize(n);
    for (int i = 0; i < n - 1; i++) {
      ds_[i] = 0.0;
      cs_[i] = 0.0;
      bs_[i] = (Y_[i + 1] - Y_[i]) / (X_[i + 1] - X_[i]);
    }
    bs_[n - 1] = bs_[n - 2];
    cs_[n - 1] = 0.0;
    ds_[n - 1] = 0.0;
  } else if (type == SplineType::CubicSpline) {
    math::BandMatrix A(n, 1, 1);
    std::vector<double> rhs(n);
    for (int i = 1; i < n - 1; i++) {
      A(i, i - 1) = 1.0 / 3.0 * (x[i] - x[i - 1]);
      A(i, i) = 2.0 / 3.0 * (x[i + 1] - x[i - 1]);
      A(i, i + 1) = 1.0 / 3.0 * (x[i + 1] - x[i]);
      rhs[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    }
    if (bc_left_ == BoundaryType::SecondDerivative) {
      A(0, 0) = 2.0;
      A(0, 1) = 0.0;
      rhs[0] = left_value_;
    } else if (bc_left_ == BoundaryType::FirstDerivative) {
      A(0, 0) = 2.0 * (x[1] - x[0]);
      A(0, 1) = 1.0 * (x[1] - x[0]);
      rhs[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - left_value_);
    } else {
      assert(false);
    }
    if (bc_right_ == BoundaryType::SecondDerivative) {
      A(n - 1, n - 1) = 2.0;
      A(n - 1, n - 2) = 0.0;
      rhs[n - 1] = right_value_;
    } else if (bc_right_ == BoundaryType::FirstDerivative) {
      A(n - 1, n - 1) = 2.0 * (x[n - 1] - x[n - 2]);
      A(n - 1, n - 2) = 1.0 * (x[n - 1] - x[n - 2]);
      rhs[n - 1] = 3.0 * (right_value_ - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
    } else {
      assert(false);
    }

    cs_ = A.lu_solve(rhs);

    ds_.resize(n);
    bs_.resize(n);
    for (int i = 0; i < n - 1; i++) {
      ds_[i] = 1.0 / 3.0 * (cs_[i + 1] - cs_[i]) / (x[i + 1] - x[i]);
      bs_[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
               1.0 / 3.0 * (2.0 * cs_[i] + cs_[i + 1]) * (x[i + 1] - x[i]);
    }
    double h = x[n - 1] - x[n - 2];
    ds_[n - 1] = 0.0;
    bs_[n - 1] = 3.0 * ds_[n - 2] * h * h + 2.0 * cs_[n - 2] * h + bs_[n - 2];
    if (bc_right_ == BoundaryType::FirstDerivative) cs_[n - 1] = 0.0;

  } else if (type == SplineType::CubicSplineHermite) {
    bs_.resize(n);
    cs_.resize(n);
    ds_.resize(n);

    for (int i = 1; i < n - 1; i++) {
      const double h = X_[i + 1] - X_[i];
      const double hl = X_[i] - X_[i - 1];
      bs_[i] = -h / (hl * (hl + h)) * Y_[i - 1] + (h - hl) / (hl * h) * Y_[i] +
               hl / (h * (hl + h)) * Y_[i + 1];
    }

    if (bc_left_ == BoundaryType::FirstDerivative) {
      bs_[0] = left_value_;
    } else if (bc_left_ == BoundaryType::SecondDerivative) {
      const double h = X_[1] - X_[0];
      bs_[0] = 0.5 * (-bs_[1] - 0.5 * left_value_ * h + 3.0 * (Y_[1] - Y_[0]) / h);
    } else {
      assert(false);
    }
    if (bc_right_ == BoundaryType::FirstDerivative) {
      bs_[n - 1] = right_value_;
      cs_[n - 1] = 0.0;
    } else if (bc_right_ == BoundaryType::SecondDerivative) {
      const double h = X_[n - 1] - X_[n - 2];
      bs_[n - 1] = 0.5 * (-bs_[n - 2] + 0.5 * right_value_ * h + 3.0 * (Y_[n - 1] - Y_[n - 2]) / h);
      cs_[n - 1] = 0.5 * right_value_;
    } else {
      assert(false);
    }
    ds_[n - 1] = 0.0;
    set_coeffs_from_b();

  } else {
    assert(false);
  }

  c0_ = (bc_left_ == BoundaryType::FirstDerivative) ? 0.0 : cs_[0];
}

bool Spline::make_monotonic() {
  assert(X_.size() == Y_.size());
  assert(X_.size() == bs_.size());
  assert(X_.size() > 2);
  bool modified = false;
  const int n = static_cast<int>(X_.size());

  for (int i = 0; i < n; i++) {
    int im1 = std::max(i - 1, 0);
    int ip1 = std::min(i + 1, n - 1);
    if (((Y_[im1] <= Y_[i]) && (Y_[i] <= Y_[ip1]) && bs_[i] < 0.0) ||
        ((Y_[im1] >= Y_[i]) && (Y_[i] >= Y_[ip1]) && bs_[i] > 0.0)) {
      modified = true;
      bs_[i] = 0.0;
    }
  }

  for (int i = 0; i < n - 1; i++) {
    double h = X_[i + 1] - X_[i];
    double avg = (Y_[i + 1] - Y_[i]) / h;
    if (avg == 0.0 && (bs_[i] != 0.0 || bs_[i + 1] != 0.0)) {
      modified = true;
      bs_[i] = 0.0;
      bs_[i + 1] = 0.0;
    } else if ((bs_[i] >= 0.0 && bs_[i + 1] >= 0.0 && avg > 0.0) ||
               (bs_[i] <= 0.0 && bs_[i + 1] <= 0.0 && avg < 0.0)) {
      double r = sqrt(bs_[i] * bs_[i] + bs_[i + 1] * bs_[i + 1]) / std::fabs(avg);
      if (r > 3.0) {
        modified = true;
        bs_[i] *= (3.0 / r);
        bs_[i + 1] *= (3.0 / r);
      }
    }
  }

  if (modified == true) {
    set_coeffs_from_b();
    monotonic_ = true;
  }

  return modified;
}

size_t Spline::find_closest(const double& x) const {
  auto it = std::upper_bound(X_.begin(), X_.end(), x);
  size_t idx = std::max(static_cast<int>(it - X_.begin()) - 1, 0);
  return idx;
}

double Spline::operator()(const double& x) const {
  size_t n = X_.size();
  size_t idx = find_closest(x);
  double h = x - X_[idx];
  double interpol;

  if (x < X_[0]) {
    interpol = (c0_ * h + bs_[0]) * h + Y_[0];
  } else if (x > X_[n - 1]) {
    interpol = (cs_[n - 1] * h + bs_[n - 1]) * h + Y_[n - 1];
  } else {
    interpol = ((ds_[idx] * h + cs_[idx]) * h + bs_[idx]) * h + Y_[idx];
  }

  return interpol;
}

std::vector<double> Spline::operator()(const std::vector<double>& X) const {
  std::vector<double> Y;
  Y.reserve(X.size());  // Reserve space to avoid reallocations

  for (auto x : X) {
    Y.push_back((*this)(x));
  }

  return Y;
}

double Spline::deriv(int order, const double& x) const {
  assert(order > 0);
  size_t n = X_.size();
  size_t idx = find_closest(x);
  double h = x - X_[idx];
  double interpol;

  if (x < X_[0]) {
    switch (order) {
      case 1:
        interpol = 2.0 * c0_ * h + bs_[0];
        break;
      case 2:
        interpol = 2.0 * c0_;
        break;
      default:
        interpol = 0.0;
        break;
    }
  } else if (x > X_[n - 1]) {
    switch (order) {
      case 1:
        interpol = 2.0 * cs_[n - 1] * h + bs_[n - 1];
        break;
      case 2:
        interpol = 2.0 * cs_[n - 1];
        break;
      default:
        interpol = 0.0;
        break;
    }
  } else {
    switch (order) {
      case 1:
        interpol = (3.0 * ds_[idx] * h + 2.0 * cs_[idx]) * h + bs_[idx];
        break;
      case 2:
        interpol = 6.0 * ds_[idx] * h + 2.0 * cs_[idx];
        break;
      case 3:
        interpol = 6.0 * ds_[idx];
        break;
      default:
        interpol = 0.0;
        break;
    }
  }

  return interpol;
}
}  // namespace math
