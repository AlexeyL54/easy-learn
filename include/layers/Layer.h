#ifndef LAYER_H
#define LAYER_H

#include <vector>

using std::vector;

/*
 * @brief Layer template implementing main operations of layer
 */
class Layer {

public:
  virtual ~Layer() = default;

  /*
   * @brief Perform forward propagation
   * @param input output data (axon signals) from previous neurons
   * @return output data of this layer
   */
  virtual vector<double> forward(const vector<double> &input) = 0;

  /*
   * @brief Perform backward propagation (adjust weights)
   * @param output_grads gradients from previous layers
   * @param learning_rate learning rate
   * @return gradient
   */
  virtual vector<double> backward(const vector<double> &output_grads) = 0;
  /*
   * @brief Save weights to a file
   */
  virtual void saveParams() = 0;

  /*
   * @brief Initialize weights with downloaded parameters form a file
   */
  virtual void downloadParams() = 0;

  /*
   * @brief Get weight values in the layer
   * @return weights
   */
  virtual vector<vector<double>> getWeights() const = 0;

  /*
   * @brief Get bias values in the layer
   * @return biases
   */
  virtual vector<double> getBiases() const = 0;

  /*
   * @brief Get weight gradient values of the layer
   * @return weight gradients
   */
  virtual vector<vector<double>> &getWeightGrads() = 0;

  /*
   * @brief Get bias gradient values of the layer
   * @return bias gradients
   */
  virtual vector<double> &getBiasGrads() = 0;

  /*
   * @brief Set new values for weights
   * @param new_weights new values of weights
   */
  virtual void setWeights(const vector<vector<double>> &new_weights) = 0;

  /*
   * @brief Set new values for biases
   * @param new_biases new values of biases
   */
  virtual void setBiases(const vector<double> &new_biases) = 0;

  /*
   * @brief Get the number of input connections
   * @return number of input connections
   */
  virtual int getInputSize() const = 0;

  /*
   * @brief Get the number of output connections
   * @return number of output connections
   */
  virtual int getOutputSize() const = 0;
};

#endif // !LAYER_H
