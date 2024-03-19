#include "json/json.hpp"

int main() {
  json::Parser parser("test.json");

  parser.parseFile();
  return 0;
}