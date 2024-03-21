#include "lstm/lstm.hpp"
#include <vector>
namespace lstm {
LSTM::LSTM(const int& input_size, const int& output_size, const int& hidden_size,
           const double& learning_rate, const double& gradient_clip)
    : input_size(input_size),
      output_size(output_size),
      hidden_size(hidden_size),
      learning_rate(learning_rate),
      gradient_clip(gradient_clip) {
  initialize_parameters();
  h = std::vector<double>(hidden_size, 0.0);
  c = std::vector<double>(hidden_size, 0.0);
  prev_h = std::vector<double>(hidden_size, 0.0);
  prev_c = std::vector<double>(hidden_size, 0.0);
}

void LSTM::initialize_parameters() {
  // Initialize parameters (weights and biases) randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normal_dist(0.0, 0.1);

  // Initialize parameters for input gate
  W_i = random_vector(hidden_size * input_size, normal_dist, gen);
  U_i = random_vector(hidden_size * hidden_size, normal_dist, gen);
  b_i = random_vector(hidden_size, normal_dist, gen);

  // Initialize parameters for forget gate
  W_f = random_vector(hidden_size * input_size, normal_dist, gen);
  U_f = random_vector(hidden_size * hidden_size, normal_dist, gen);
  b_f = random_vector(hidden_size, normal_dist, gen);

  // Initialize parameters for cell gate
  W_c = random_vector(hidden_size * input_size, normal_dist, gen);
  U_c = random_vector(hidden_size * hidden_size, normal_dist, gen);
  b_c = random_vector(hidden_size, normal_dist, gen);

  // Initialize parameters for output gate
  W_o = random_vector(hidden_size * input_size, normal_dist, gen);
  U_o = random_vector(hidden_size * hidden_size, normal_dist, gen);
  b_o = random_vector(hidden_size, normal_dist, gen);

  // Additional parameters for output layer
  V = random_vector(output_size * hidden_size, normal_dist, gen);
  c_out = random_vector(output_size, normal_dist, gen);
}

std::vector<double> LSTM::forward(const std::vector<double>& input) {
  // Compute input gate output
  std::vector<double> input_gate_input =
      add_matrices(dot_product(W_i, input), dot_product(U_i, prev_h));
  std::vector<double> input_gate_activation = sigmoid(add_matrices(input_gate_input, b_i));

  // Compute forget gate output
  std::vector<double> forget_gate_input =
      add_matrices(dot_product(W_f, input), dot_product(U_f, prev_h));
  std::vector<double> forget_gate_activation = sigmoid(add_matrices(forget_gate_input, b_f));

  // Compute cell gate activation output
  std::vector<double> cell_gate_input =
      add_matrices(dot_product(W_c, input), dot_product(U_c, prev_h));
  std::vector<double> cell_gate_activation = tanh(add_matrices(cell_gate_input, b_c));

  // Compute cell gate output
  std::vector<double> cell_gate_output =
      elementwise_product(forget_gate_activation, prev_c) +
      elementwise_product(input_gate_activation, cell_gate_activation);

  // Compute output gate output
  std::vector<double> output_gate_input =
      add_matrices(dot_product(W_o, input), dot_product(U_o, prev_h));
  std::vector<double> output_gate_activation = sigmoid(add_matrices(output_gate_input, b_o));

  // Compute hidden activation output
  std::vector<double> hidden_activation_output =
      elementwise_product(output_gate_activation, tanh(cell_gate_output));

  // Update previous hidden and cell states
  prev_h = hidden_activation_output;
  prev_c = cell_gate_output;

  return hidden_activation_output;
}

void LSTM::backward(std::vector<double>& input, std::vector<double>& dh_next,
                    std::vector<double>& dc_next) {
  std::vector<double> dh = dh_next;
  std::vector<double> dc = dc_next;

  std::vector<double> dout = dh;
  std::vector<double> dhidden_activation =
      elementwise_product(dout, tanh_derivative(hidden_activation_output));
  std::vector<double> doutput_gate = elementwise_product(dout, tanh(cell_gate_output));
  std::vector<double> doutput_gate_input = sigmoid_derivative(output_gate_output);
  doutput_gate = elementwise_product(doutput_gate, doutput_gate_input);
  std::vector<double> dcell_gate_output = elementwise_product(dout, output_gate_output);
  std::vector<double> dcell_gate_activation_output = tanh_derivative(cell_gate_activation_output);
  std::vector<double> dcell_gate =
      elementwise_product(dcell_gate_output, dcell_gate_activation_output);
  std::vector<double> dforget_gate_output = elementwise_product(dcell_gate_output, prev_c);
  std::vector<double> dforget_gate_input = sigmoid_derivative(forget_gate_output);
  std::vector<double> dforget_gate = elementwise_product(dforget_gate_output, dforget_gate_input);
  std::vector<double> dinput_gate_output =
      elementwise_product(dcell_gate_output, cell_gate_activation_output);
  std::vector<double> dinput_gate_input = sigmoid_derivative(input_gate_output);
  std::vector<double> dinput_gate = elementwise_product(dinput_gate_output, dinput_gate_input);

  input_gate_grad = dot_product(dinput_gate, input);
  forget_gate_grad = dot_product(dforget_gate, input);
  cell_gate_grad = dot_product(dcell_gate, input);
  output_gate_grad = dot_product(doutput_gate, input);

  std::vector<double> dprev_h = dot_product(W_i, dinput_gate) + dot_product(W_f, dforget_gate) +
                                dot_product(W_c, dcell_gate) + dot_product(W_o, doutput_gate);
  std::vector<double> dprev_c = elementwise_product(dcell_gate, forget_gate_output) +
                                elementwise_product(dh, output_gate_output);

  clip_gradients(input_gate_grad);
  clip_gradients(forget_gate_grad);
  clip_gradients(cell_gate_grad);
  clip_gradients(output_gate_grad);

  update_parameters();

  dh_next = dprev_h;
  dc_next = dprev_c;
}

void LSTM::update_parameters() {
  W_i = elementwise_subtraction(W_i, scalar_multiplication(learning_rate, input_gate_grad));
  W_f = elementwise_subtraction(W_f, scalar_multiplication(learning_rate, forget_gate_grad));
  W_c = elementwise_subtraction(W_c, scalar_multiplication(learning_rate, cell_gate_grad));
  W_o = elementwise_subtraction(W_o, scalar_multiplication(learning_rate, output_gate_grad));

  U_i = elementwise_subtraction(U_i, scalar_multiplication(learning_rate, input_gate_grad));
  U_f = elementwise_subtraction(U_f, scalar_multiplication(learning_rate, forget_gate_grad));
  U_c = elementwise_subtraction(U_c, scalar_multiplication(learning_rate, cell_gate_grad));
  U_o = elementwise_subtraction(U_o, scalar_multiplication(learning_rate, output_gate_grad));

  b_i = elementwise_subtraction(b_i, scalar_multiplication(learning_rate, input_gate_grad));
  b_f = elementwise_subtraction(b_f, scalar_multiplication(learning_rate, forget_gate_grad));
  b_c = elementwise_subtraction(b_c, scalar_multiplication(learning_rate, cell_gate_grad));
  b_o = elementwise_subtraction(b_o, scalar_multiplication(learning_rate, output_gate_grad));
}

void LSTM::clip_gradients(std::vector<double>& gradients) {
  for (auto& grad : gradients) {
    grad = std::min(grad, gradient_clip);
    grad = std::max(grad, -gradient_clip);
  }
}

std::vector<double> LSTM::sigmoid(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (int i = 0; i < x.size(); ++i) {
    result[i] = 1.0 / (1.0 + exp(-x[i]));
  }
  return result;
}

std::vector<double> LSTM::sigmoid_derivative(const std::vector<double>& x) {
  std::vector<double> sig = sigmoid(x);
  std::vector<double> result(x.size());
  for (int i = 0; i < x.size(); ++i) {
    result[i] = sig[i] * (1 - sig[i]);
  }
  return result;
}

std::vector<double> LSTM::tanh(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (int i = 0; i < x.size(); ++i) {
    result[i] = std::tanh(x[i]);
  }
  return result;
}

std::vector<double> LSTM::tanh_derivative(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (int i = 0; i < x.size(); ++i) {
    result[i] = 1.0 - x[i] * x[i];
  }
  return result;
}

std::vector<double> LSTM::elementwise_product(const std::vector<double>& a,
                                              const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (int i = 0; i < a.size(); ++i) {
    result[i] = a[i] * b[i];
  }
  return result;
}

std::vector<double> LSTM::elementwise_subtraction(const std::vector<double>& a,
                                                  const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (int i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> LSTM::scalar_multiplication(double scalar, const std::vector<double>& vector) {
  std::vector<double> result(vector.size());
  for (int i = 0; i < vector.size(); ++i) {
    result[i] = scalar * vector[i];
  }
  return result;
}

std::vector<double> LSTM::add_matrices(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (int i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> LSTM::dot_product(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> result(hidden_size, 0.0);
  for (int i = 0; i < hidden_size; ++i) {
    for (int j = 0; j < input_size; ++j) {
      result[i] += a[i * input_size + j] * b[j];
    }
  }
  return result;
}

std::vector<double> LSTM::random_vector(int size, std::normal_distribution<>& distribution,
                                        std::mt19937& generator) {
  std::vector<double> result(size);
  for (int i = 0; i < size; ++i) {
    result[i] = distribution(generator);
  }
  return result;
}

std::vector<double> LSTM::get_hidden_state() { return h; }

std::vector<double> LSTM::get_cell_state() { return c; }

}  // namespace lstm