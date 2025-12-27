#include "../include/SequentialModel.h"
#include <iostream>
#include <memory>
#include <vector>

using std::vector;

/*
 * @brief Add a layer to the model
 * @param layer pointer to a layer object
 */
void SequentialModel::addLayer(std::unique_ptr<Layer> layer) {
  layers.push_back(std::move(layer));
}

/*
 * @brief Get the model's output (prediction)
 * @param input input data (features)
 * @return output value
 */
vector<double> SequentialModel::predict(const vector<double> &input) {
  vector<double> activation = input;

  for (std::unique_ptr<Layer> &layer : layers) {
    activation = layer->forward(activation);
  }
  return activation;
}

/*
 * @brief Train the model
 * @param input input data (features)
 * @param target expected output data
 * @param learning_rate learning rate
 * @return error
 */
double SequentialModel::train(const vector<double> &input,
                              const vector<double> &target,
                              const double learning_rate) {
  double loss = 0.0;
  double error = 0.0;
  vector<double> output = predict(input);
  vector<double> gradient(output.size());

  // Calculate MSE
  for (size_t i = 0; i < output.size(); i++) {
    error = output[i] - target[i];
    loss += error * error;
  }
  loss /= output.size();

  // Calculate gradient
  for (size_t i = 0; i < output.size(); i++) {
    gradient[i] = 2.0 * (output[i] - target[i]) / output.size();
  }

  // Perform back propagation
  for (std::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it) {
    gradient = (*it)->backward(gradient, learning_rate);
  }
  return loss;
}

/*
 * @brief Perform one epoch of training
 * @param inputs input data (features)
 * @param targets reference output values
 * @param learning_rate learning rate
 * @param verbose flag to output error information
 */
void SequentialModel::train_epoch(const vector<vector<double>> &inputs,
                                  const vector<vector<double>> &targets,
                                  double learning_rate, bool verbose) {
  double total_loss = 0.0;

  for (size_t i = 0; i < inputs.size(); i++) {
    total_loss += train(inputs[i], targets[i], learning_rate);
  }

  if (verbose) {
    std::cout << "Everage error for epoch: " << total_loss / inputs.size()
              << std::endl;
  }
}

/*
 * @brief Save weights of each layer
 */
void SequentialModel::saveParams() {
  for (std::unique_ptr<Layer> &layer : layers) {
    layer->saveParams();
  }
}

/*
 * @brief Initialize each layers weights in model with downloaded parameters
 */
void SequentialModel::downloadParams() {
  for (std::unique_ptr<Layer> &layer : layers) {
    layer->downloadParams();
  }
};
