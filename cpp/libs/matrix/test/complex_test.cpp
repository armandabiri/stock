#include "matrix/complex.hpp"

int main() {
  math::Complex a(1, 2);
  math::Complex b(3, 4);

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "a + b: " << a + b << std::endl;
  std::cout << "a - b: " << a - b << std::endl;
  std::cout << "a * b: " << a * b << std::endl;
  std::cout << "a / b: " << a / b << std::endl;
  std::cout << "a + 1: " << a + 1 << std::endl;
  std::cout << "1 + a: " << 1 + a << std::endl;
  std::cout << "a - 1: " << a - 1 << std::endl;
  std::cout << "1 - a: " << 1 - a << std::endl;
  std::cout << "a * 2: " << a * 2 << std::endl;
  std::cout << "2 * a: " << 2 * a << std::endl;
  std::cout << "a / 2: " << a / 2 << std::endl;
  std::cout << "2 / a: " << 2 / a << std::endl;
  std::cout << "a.norm(): " << a.norm() << std::endl;
  std::cout << "a.conj(): " << a.conj() << std::endl;

  std::cout << "a += b: " << (a += b) << std::endl;
  std::cout << "a -= b: " << (a -= b) << std::endl;
  std::cout << "a *= b: " << (a *= b) << std::endl;
  std::cout << "a /= b: " << (a /= b) << std::endl;

  std::cout << "exp(1 + 1 * math::i)= " << exp(1 + 1 * math::i1) << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "exp(a)= " << exp(a) << std::endl;
  std::cout << "exp(a + b)= " << exp(a + b) << std::endl;

  return 0;
}