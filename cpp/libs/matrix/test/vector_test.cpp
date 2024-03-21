#include "matrix/vector.hpp"

int main() {
  matrix::Vector a(3, 1);  // Create a row vector of size 3
  matrix::Vector b(3, 2);  // Create a row vector of size 3
  a(2) = 10;

  std::cout << "a(2)=" << a(2) << std::endl;
  std::cout << "a + b=" << a + b << std::endl;  // Print the sum of the two vectors std::cout
  std::cout << "a - b=" << a - b << std::endl;  // Print the difference of the two vectors
  std::cout << "a * 2=" << a * 2 << std::endl;  // Print the product of the vector and a scalar
  std::cout << "a / 2=" << a / 2 << std::endl;  // Print the division of the vector by a scalar
  std::cout << "a .* b=" << a.elementwiseProd(b)
            << std::endl;  // Print the element-wise product of the two vectors
  std::cout << "a.norm=" << a.norm()
            << std::endl;  // Print the element-wise division of the two vectors
  std::cout << "a.dot(b)=" << a.dot(b) << std::endl;
  // std::cout << "a.cross(b)=" << a.cross(b) << std::endl;
  std::cout << "a^T*b=" << a.transpose() * b << std::endl;

  std::cout << "a*b^T=" << a * (b.transpose()) << std::endl;

  matrix::Vector c = a + b;
  c = a.transpose() * b;
  //   std::cout << c;  // Print the dot product of the two vectors

  return 0;
}