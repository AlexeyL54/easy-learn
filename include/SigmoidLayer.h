#pragma once

#include "Layer.h"
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

/*
 * @brief Implementation of a layer with the sigmoid activation function
 */
class SigmoidLayer : public Layer {

private:
  vector<vector<double>>
      weights;                // weights for each input of each neuron in layer
  vector<double> biases;      // biases for each neuron
  vector<double> last_input;  // last input data
  vector<double> last_output; // last output data
  vector<double> last_z;      // weighted sum
  int input_size;             // size of input data
  int output_size;            // number of neurons in layer
  std::string config_name;    // path of file to save weights

public:
  SigmoidLayer(int input, int neurons, std::string file_name);

  /*
   * @brief Perform forward propagation
   * @param input output data (axon signals) from previous neurons
   * @return output data of this layer
   */
  vector<double> forward(const vector<double> &input) override;

  /*
   * @brief Perform backward propagation (adjust weights)
   * @param output_grads gradients from previous layers
   * @param learning_rate learning rate
   * @return gradient
   */
  vector<double> backward(const vector<double> &output_gradient,
                          double learning_rate) override;

  /*
   * @brief Save weights to a file
   */
  void saveParams() override;

  /*
   * @brief Initialize weights with download parameters form a file
   */
  void downloadParams() override;

  /*
   * @brief Get weight values in the layer
   * @return weights
   */
  vector<vector<double>> getWeights() const override;

  /*
   * @brief Set new values for weights
   */
  void setWeights(const vector<vector<double>> &new_weights) override;

  /*
   * @brief Get the number of input connections
   * @return number of input connections
   */
  int getInputSize() const override;

  /*
   * @brief Get the number of output connections
   * @return number of output connections
   */
  int getOutputSize() const override;
};
