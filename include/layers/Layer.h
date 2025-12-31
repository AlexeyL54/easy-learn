#ifndef LAYER_HPP
#define LAYER_HPP

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
  virtual vector<double> backward(const vector<double> &output_grads,
                                  double learning_rate) = 0;
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
   * @brief Set new values for weights
   * @param new_weights new values of weights
   */
  virtual void setWeights(const vector<vector<double>> &new_weights) = 0;

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

#endif
