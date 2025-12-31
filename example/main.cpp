#include "../include/SequentialModel.h"
#include "../include/layers/SigmoidLayer.h"
#include "../include/loss/MSE.h"
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

void sigmoidNetworkExample() {
  std::cout << "=== Sigmoid net for XOR ===" << std::endl;

  std::vector<std::unique_ptr<Layer>> layers;
  layers.emplace_back(std::make_unique<SigmoidLayer>(2, 4, "layer1.txt"));
  layers.emplace_back(std::make_unique<SigmoidLayer>(4, 1, "layer2.txt"));

  SequentialModel model(std::move(layers), std::make_unique<MSE>(), 1000, 0.5);

  model.train(inputs, targets);

  std::cout << "Results:" << std::endl;
  for (size_t i = 0; i < inputs.size(); i++) {
    vector<double> prediction = model.predict(inputs[i]);
    std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = "
              << prediction[0] << " (expected: " << targets[i][0] << ")"
              << std::endl;
  }
}

int main() {
  sigmoidNetworkExample();
  return 0;
}
