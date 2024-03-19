#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

inline std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vectors must have the same size for addition");
  }

  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

namespace lstm {

class LSTM {
 public:
  LSTM(const int& input_size, const int& output_size, const int& hidden_size,
       const double& learning_rate, const double& gradient_clip);

  void initialize_parameters();
  std::vector<double> forward(const std::vector<double>& input);
  void backward(std::vector<double>& input, std::vector<double>& dh_next,
                std::vector<double>& dc_next);
  void update_parameters();
  void clip_gradients(std::vector<double>& gradients);

  std::vector<double> sigmoid(const std::vector<double>& x);
  std::vector<double> sigmoid_derivative(const std::vector<double>& x);
  std::vector<double> tanh(const std::vector<double>& x);
  std::vector<double> tanh_derivative(const std::vector<double>& x);

  std::vector<double> elementwise_product(const std::vector<double>& a,
                                          const std::vector<double>& b);
  std::vector<double> elementwise_subtraction(const std::vector<double>& a,
                                              const std::vector<double>& b);
  std::vector<double> scalar_multiplication(double scalar, const std::vector<double>& vector);
  std::vector<double> dot_product(const std::vector<double>& a, const std::vector<double>& b);
  std::vector<double> random_vector(int size, std::normal_distribution<>& distribution,
                                    std::mt19937& generator);
  std::vector<double> add_matrices(const std::vector<double>& a, const std::vector<double>& b);

  std::vector<double> get_hidden_state();
  std::vector<double> get_cell_state();
  inline void increment_epoch() { epoch++; }

 private:
  int input_size, output_size;
  int hidden_size;
  double learning_rate;
  double gradient_clip;
  int epoch;

  std::vector<double> W_i, W_f, W_c, W_o;
  std::vector<double> U_i, U_f, U_c, U_o;
  std::vector<double> b_i, b_f, b_c, b_o;
  std::vector<double> V, c_out;

  std::vector<double> h, c;
  std::vector<double> prev_h, prev_c;

  std::vector<double> input_gate_output, forget_gate_output, cell_gate_output, output_gate_output;
  std::vector<double> cell_gate_activation_output, hidden_activation_output;
  std::vector<double> input_gate_grad, forget_gate_grad, cell_gate_grad, output_gate_grad;
  std::vector<double> cell_gate_activation_grad, hidden_activation_grad;
};

}  // namespace lstm