#pragma once

#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>  // For debugging, you can remove this later

namespace json {
class Parser {
 private:
  static const int MAX_LENGTH = 100;
  std::unordered_map<std::string, void*> objects;  // Use actual object, not pointer
  std::ifstream jsonFile;

 public:
  Parser(const std::string& path) {
    jsonFile.open(path);  // No need to specify ifstream::in, it's the default mode
    if (!jsonFile.is_open()) {
      std::cerr << "Failed to open file: " << path << std::endl;
      return;
    }
    parseFile();
  }

  void parseFile() {
    std::string line;
    while (std::getline(jsonFile, line)) {  // Read line by line
      std::vector<std::string> tokens = lexer(line);
      // Now, you can do something with tokens
      // For debugging, let's print tokens
      for (const auto& token : tokens) {
        std::cout << token << std::endl;
      }
    }
  }

  std::vector<std::string> lexer(const std::string& inputLine) {
    std::vector<std::string> tokens;
    std::string token;
    for (char character : inputLine) {
      if (character == '{' || character == '}' || character == ':' || character == '[' ||
          character == ']') {
        if (!token.empty()) {
          tokens.push_back(token);
          token.clear();
        }
        tokens.push_back(std::string(1, character));  // Convert char to string
      } else if (character != ' ') {
        token += character;
      }
    }
    if (!token.empty()) {
      tokens.push_back(token);
    }
    return tokens;
  }

  ~Parser() { jsonFile.close(); }
};
}  // namespace json
