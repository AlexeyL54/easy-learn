#include "../include/ReLULayer.h"
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

ReLULayer::ReLULayer(int input, int neurons, std::string file_name) {
  input_size = input;
  output_size = neurons;
  config_name = file_name;

  // He weights initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  double stddev = std::sqrt(2.0 / input_size);
  std::normal_distribution<double> dist(0.0, stddev);

  weights.resize(output_size, std::vector<double>(input_size));
  biases.resize(output_size, 0.1);

  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      weights[i][j] = dist(gen);
    }
  }
}

/*
 * @brief Perform forward propagation
 * @param input output data (axon signals) from previous neurons
 * @return output data of this layer
 */
vector<double> ReLULayer::forward(const std::vector<double> &input) {
  last_input = input;
  int output_size = weights.size();
  vector<double> output(output_size);
  last_z.resize(output_size);

  // for each output
  for (int i = 0; i < output_size; i++) {
    last_z[i] = biases[i];

    // for each input
    for (size_t j = 0; j < input.size(); j++) {
      last_z[i] += input[j] * weights[i][j];
    }

    // ReLU activation
    output[i] = std::max(0.0, last_z[i]);
  }

  last_output = output;
  return output;
}

/*
 * @brief Perform backward propagation (adjust weights)
 * @param output_grads gradients from previous layers
 * @param learning_rate learning rate
 * @return gradient
 */
vector<double> ReLULayer::backward(const std::vector<double> &output_gradient,
                                   double learning_rate) {
  int input_size = last_input.size();
  int output_size = weights.size();
  std::vector<double> input_gradient(input_size, 0.0);

  // Compute gradient with respect to the weighted sum (z)
  std::vector<double> z_gradient(output_size);
  for (int i = 0; i < output_size; i++) {
    double activation_derivative = last_z[i] > 0 ? 1.0 : 0.0;
    z_gradient[i] = output_gradient[i] * activation_derivative;
  }

  // Compute gradients for weights and update them
  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      double weight_gradient = z_gradient[i] * last_input[j];
      weights[i][j] -= learning_rate * weight_gradient;
      input_gradient[j] += z_gradient[i] * weights[i][j];
    }
    biases[i] -= learning_rate * z_gradient[i];
  }

  return input_gradient;
}

/*
 * @brief Save weights to a file
 */
void ReLULayer::saveParams() {
  std::ofstream file(config_name);

  if (file.is_open()) {

    file << input_size << "\n";
    file << output_size << "\n";

    for (vector<double> &vec : weights) {
      for (double weight : vec) {
        file << weight << " ";
      }
      file << "\n";
    }
    for (double &bias : biases) {
      file << bias << " ";
    }
    file.close();
  }
}

/*
 * @brief Initialize weights with download parameters form a file
 */
void ReLULayer::downloadParams() {
  std::string line;
  double value;
  std::ifstream file(config_name);

  if (file.is_open()) {
    std::getline(file, line);
    input_size = std::stoi(line);

    std::getline(file, line);
    output_size = std::stoi(line);

    weights.resize(output_size, std::vector<double>(input_size));

    // Read weights
    for (int i = 0; i < output_size; i++) {
      std::getline(file, line);
      std::stringstream s(line);
      vector<double> row_weights;

      while (s >> value) {
        row_weights.push_back(value);
      }

      // Check size
      if (row_weights.size() != static_cast<size_t>(input_size)) {
        throw std::runtime_error("Weight size mismatch in SigmoidLayer");
      }

      weights[i] = row_weights;
    }

    biases.resize(output_size);

    // Read biases
    std::getline(file, line);
    std::stringstream s(line);
    vector<double> loaded_biases;

    while (s >> value) {
      loaded_biases.push_back(value);
    }

    // Check size
    if (loaded_biases.size() != static_cast<size_t>(output_size)) {
      throw std::runtime_error("Bias size mismatch in SigmoidLayer");
    }

    biases = loaded_biases;
    file.close();
  }
}

/*
 * @brief Get weight values in the layer
 * @return weights
 */
vector<vector<double>> ReLULayer::getWeights() const { return weights; }

/*
 * @brief Set new values for weights
 */
void ReLULayer::setWeights(const vector<vector<double>> &new_weights) {
  weights = new_weights;
}

/*
 * @brief get the number of input connections
 * @return number of input connections
 */
int ReLULayer::getInputSize() const { return weights[0].size(); }

/*
 * @brief Get the number of output connections
 * @return number of output connections
 */
int ReLULayer::getOutputSize() const { return weights.size(); };
