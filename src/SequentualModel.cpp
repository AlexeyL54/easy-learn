#include "../include/SequentialModel.h"
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using std::vector;

SequentialModel::SequentialModel(vector<std::unique_ptr<Layer>> layers_vec,
                                 std::unique_ptr<Loss> loss_function,
                                 std::unique_ptr<Optimizer> optimizer,
                                 int total_epochs)

    : layers(std::move(layers_vec)), loss_func(std::move(loss_function)),
      optimizer(std::move(optimizer)), epochs(total_epochs) {}

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
 * @brief Perform back propagation
 */
void SequentialModel::backward() {
  vector<double> gradient = loss_func->computeGrad();

  for (int i = layers.size() - 1; i > 0; i--) {
    gradient = layers[i]->backward(gradient);
    optimizer->step(*layers[i]);
  }
}

/*
 * @brief Perform one epoch of training
 * @param inputs input data (features)
 * @param targets reference output values
 */
void SequentialModel::train(const vector<vector<double>> &inputs,
                            const vector<vector<double>> &targets) {
  for (int epoch = 1; epoch <= epochs; epoch++) {

    double loss = 0.0;

    for (int i = 0; i < inputs.size(); i++) {
      vector<double> output = predict(inputs[i]);
      loss += loss_func->computeLoss(output, targets[i]);
      backward();
    }

    if (epoch % (epochs / 10) == 0)
      std::cout << "Average loss after " << epoch
                << " epochs = " << loss / inputs[0].size() << std::endl;
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
