#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cassert>

#include "math/band_matrix.hpp"

namespace math {
inline std::vector<double> resample(const std::vector<double>& input, size_t m) {
  size_t n = input.size();
  if (m >= n) {
    return input;
  }

  std::vector<double> output;

  // Calculate the downsampling factor
  size_t factor = n / m;

  // Iterate through the input vector and select every 'factor'-th element
  for (size_t i = 0; i < n; i += factor) {
    output.push_back(input[i]);
  }

  return output;
}

// Spline types
enum class SplineType {
  Linear = 10,             // Linear interpolation
  CubicSpline = 30,        // Cubic splines (classical C^2)
  CubicSplineHermite = 31  // Cubic Hermite splines (local, only C^1)
};

// Boundary condition type for the Spline end-points
enum class BoundaryType { FirstDerivative = 1, SecondDerivative = 2 };

// Spline interpolation
class Spline {
 public:
  Spline()
      : spline_type_(SplineType::CubicSpline),
        bc_left_(BoundaryType::SecondDerivative),
        bc_right_(BoundaryType::SecondDerivative),
        left_value_(0.0),
        right_value_(0.0),
        monotonic_(false) {}

  // Constructor
  Spline(const std::vector<double>& X, const std::vector<double>& Y, const size_t& sampled = 0,
         SplineType type = SplineType::CubicSpline, bool make_monotonic = false,
         BoundaryType left = BoundaryType::SecondDerivative, const double& left_value = 0.0,
         BoundaryType right = BoundaryType::SecondDerivative, const double& right_value = 0.0)
      : spline_type_(type),
        bc_left_(left),
        bc_right_(right),
        left_value_(left_value),
        right_value_(right_value),
        monotonic_(false)  // false correct here: make_monotonic() sets it
  {
    if (sampled > 0) {
      std::vector<double> Xs = resample(X, sampled);
      std::vector<double> Ys = resample(Y, sampled);
      this->set_points(Xs, Ys, spline_type_);
    } else {
      // Set the data points (and optionally make the spline monotonic
      this->set_points(X, Y, spline_type_);
    }
    if (make_monotonic) {
      this->make_monotonic();
    }
  }

  // Modify boundary conditions
  void set_boundary(const BoundaryType& left, const double& left_value, const BoundaryType& right,
                    const double& right_value);

  // Set all data points
  void set_points(const std::vector<double>& x, const std::vector<double>& y,
                  const SplineType& type = SplineType::CubicSpline);

  // Adjust coefficients so that the Spline becomes piecewise monotonic
  bool make_monotonic();

  // Evaluate the Spline at point x
  double operator()(const double& x) const;
  std::vector<double> operator()(const std::vector<double>& X) const;
  double deriv(int order, const double& x) const;

  // Return the input data points
  std::vector<double> get_x() const { return X_; }
  std::vector<double> get_y() const { return Y_; }
  double get_x_min() const {
    assert(!X_.empty());
    return X_.front();
  }
  double get_x_max() const {
    assert(!X_.empty());
    return X_.back();
  }

 protected:
  std::vector<double> X_, Y_;         // x, y coordinates of points
  std::vector<double> bs_, cs_, ds_;  // Spline coefficients
  double c0_;                         // for left extrapolation
  SplineType spline_type_;
  BoundaryType bc_left_, bc_right_;
  double left_value_, right_value_;
  bool monotonic_;

  void set_coeffs_from_b();                    // calculate c_i, d_i from b_i
  size_t find_closest(const double& x) const;  // closest idx so that X_[idx]<=x
};
}  // namespace math
