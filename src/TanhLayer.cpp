#include "../include/layers/TanhLayer.h"
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

TanhLayer::TanhLayer(int input, int neurons, std::string file_name) {
  input_size = input;
  output_size = neurons;
  config_name = file_name;

  // Xavier/Glorot weights initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  double stddev = std::sqrt(2.0 / (input_size + output_size));
  std::normal_distribution<double> dist(0.0, stddev);

  weights.resize(output_size, std::vector<double>(input_size));
  weight_grads.resize(output_size, std::vector<double>(input_size));
  biases.resize(output_size, 0.1);
  bias_grads.resize(output_size, 0.1);

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
vector<double> TanhLayer::forward(const vector<double> &input) {
  last_input = input;
  int output_size = weights.size();
  vector<double> output(output_size);
  last_z.resize(output_size);

  for (size_t i = 0; i < output_size; i++) {
    last_z[i] = biases[i];

    for (size_t j = 0; j < input.size(); j++) {
      last_z[i] += weights[i][j] * input[j];
    }
    // Tanh activation
    output[i] = std::tanh(last_z[i]);
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
std::vector<double>
TanhLayer::backward(const std::vector<double> &output_gradient) {
  int input_size = last_input.size();
  int output_size = weights.size();
  std::vector<double> input_gradient(input_size, 0.0);

  // Compute gradient with respect to the weighted sum (z)
  // bias gradients are equal to z gradients here
  for (int i = 0; i < output_size; i++) {
    double activation_derivative = 1.0 - last_output[i] * last_output[i];
    bias_grads[i] = output_gradient[i] * activation_derivative;

    for (int j = 0; j < input_size; j++) {
      weight_grads[i][j] = bias_grads[i] * last_input[j];
      input_gradient[j] += bias_grads[i] * weights[i][j];
    }
  }

  return input_gradient;
}

/*
 * @brief Save weights to a file
 */
void TanhLayer::saveParams() {
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
void TanhLayer::downloadParams() {
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
        throw std::runtime_error("Weight size mismatch in TanhLayer");
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
      throw std::runtime_error("Bias size mismatch in TanhLayer");
    }

    biases = loaded_biases;
    file.close();
  }
}

/*
 * @brief Get weight values in the layer
 * @return weights
 */
vector<vector<double>> TanhLayer::getWeights() const { return weights; }

/*
 * @brief Get bias values in the layer
 * @return biases
 */
vector<double> TanhLayer::getBiases() const { return biases; }

/*
 * @brief Get weight gradient values of the layer
 * @return weight gradients
 */
vector<vector<double>> &TanhLayer::getWeightGrads() { return weight_grads; };

/*
 * @brief Get bias gradient values of the layer
 * @return bias gradients
 */
vector<double> &TanhLayer::getBiasGrads() { return bias_grads; };

/*
 * @brief Set new values for weights
 */
void TanhLayer::setWeights(const vector<vector<double>> &new_weights) {
  weights = new_weights;
}

/*
 * @brief Set new values for biases
 * @param new_biases new values of biases
 */
void TanhLayer::setBiases(const vector<double> &new_biases) {
  biases = new_biases;
}

/*
 * @brief get the number of input connections
 * @return number of input connections
 */
int TanhLayer::getInputSize() const { return weights[0].size(); }

/*
 * @brief Get the number of output connections
 * @return number of output connections
 */
int TanhLayer::getOutputSize() const { return weights.size(); };
