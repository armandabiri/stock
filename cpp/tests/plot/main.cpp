#include <cmath>
#include <matplot/matplot.h>

int main() {
  std::vector<double> x = matplot::linspace(0, 2 * matplot::pi, 100);
  std::vector<double> y = matplot::transform(x, [](auto x) { return sin(x); });

  matplot::hold(matplot::on);
  matplot::plot(x, y);

  matplot::title("Some functions of $x$");  // add a title
  matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
  matplot::plot(matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), y,
                "-:gs");  // Corrected line
  matplot::xlabel("x");
  matplot::show();
  return 0;
}
