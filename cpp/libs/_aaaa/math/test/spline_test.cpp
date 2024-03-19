#include "math/spline.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <matplot/matplot.h>
#include <cmath>
#include <random>

std::vector<double> linspace(double start, double end, int num) {
  std::vector<double> result(num);
  double step = (end - start) / (num - 1);
  for (int i = 0; i < num; ++i) {
    result[i] = start + i * step;
  }
  return result;
}

constexpr long double pi = 3.141592653589793238462643383279502884L;

int main() {
  // Generate Y values based on y = x^2 using std::transform
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);  // Random numbers between -1 and 1

  std::vector<double> X = linspace(0.0, 2 * pi, 400);
  std::vector<double> Xi = linspace(0.0, 2 * pi, 100);

  std::vector<double> Y(X.size());

  // Generate Y values based on y = x^2 using std::transform
  std::transform(X.begin(), X.end(), Y.begin(), [&dis, &gen](double x) {
    return 10 * exp(-.01 * abs(dis(gen))) * std::sin(x) + dis(gen) * std::sin(1000 * dis(gen) * x);
  });

  matplot::hold(matplot::on);
  matplot::plot(X, Y, "-k");

  math::Spline s(X, Y, 20);
  double x = 1, y = s(x), deriv = s.deriv(1, x);

  printf("spline at %f is %f with 1st derivative %f and 2nd derivative %f\n", x, y, s.deriv(1, x),
         s.deriv(2, x));

  matplot::plot(Xi, s(Xi), "-r");
  matplot::show();
}