#pragma once

#include "layers/Layer.h"
#include "loss/Loss.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

using std::vector;

/*
 * @brief Implements building a model from layers, training, and prediction
 */
class SequentialModel {
private:
  vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<Loss> loss_func;
  int epochs;
  float lr;

public:
  SequentialModel(vector<std::unique_ptr<Layer>> layers,
                  std::unique_ptr<Loss> loss_function, int total_epochs,
                  float learning_rate);

  /*
   * @brief Add a layer to the model
   * @param layer pointer to a layer object
   */
  void addLayer(std::unique_ptr<Layer> layer);

  /*
   * @brief Get the model's output (prediction)
   * @param input input data (features)
   * @return output value
   */
  vector<double> predict(const vector<double> &input);

  /*
   * @brief Perform back propagation
   */
  void backward();

  /*
   * @brief Train the model
   * @param input input data (features)
   * @param target expected output data
   * @param learning_rate learning rate
   * @return error
   */
  void train(const vector<vector<double>> &inputs,
             const vector<vector<double>> &targets);

  /*
   * @brief Perform one epoch of training
   * @param inputs input data (features)
   * @param targets reference output values
   * @param learning_rate learning rate
   * @param verbose flag to output error information
   */
  void train_epoch(const vector<vector<double>> &inputs,
                   const vector<vector<double>> &targets, double learning_rate,
                   bool verbose = false);

  /*
   * @brief Save weights of each layer
   */
  void saveParams();

  /*
   * @brief Initialize each layers weights in model with downloaded parameters
   */
  void downloadParams();
};
