#include "../include/optimizers/SGD.h"

SGD::SGD(float lr) : learning_rate(lr) {}

/*
 * @brief Correct weights
 * @param layer pointer to the layer object
 */
void SGD::step(Layer &layer) {
  int input_size = layer.getInputSize();
  int output_size = layer.getOutputSize();

  vector<vector<double>> weights = layer.getWeights();
  vector<double> biases = layer.getBiases();

  vector<vector<double>> &weight_grads = layer.getWeightGrads();
  vector<double> &biase_grads = layer.getBiasGrads();

  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      weights[i][j] -= learning_rate * weight_grads[i][j];
    }
    biases[i] -= learning_rate * biase_grads[i];
  }

  layer.setWeights(weights);
  layer.setBiases(biases);
}
