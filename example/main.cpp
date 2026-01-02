#include "../include/SequentialModel.h"

#include "../include/layers/ReLULayer.h"
#include "../include/layers/SigmoidLayer.h"
#include "../include/layers/TanhLayer.h"

#include "../include/loss/MSE.h"
#include "../include/optimizers/SGD.h"

#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

void sigmoidNetworkExample() {
  std::cout << "=== Sigmoid net for XOR ===" << std::endl;

  std::vector<std::unique_ptr<Layer>> layers;
  layers.emplace_back(std::make_unique<ReLULayer>(2, 8, "layer1.txt"));
  layers.emplace_back(std::make_unique<TanhLayer>(8, 4, "layer2.txt"));
  layers.emplace_back(std::make_unique<SigmoidLayer>(4, 1, "layer3.txt"));

  SequentialModel model(std::move(layers), std::make_unique<MSE>(),
                        std::make_unique<SGD>(0.1), 1000);

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
