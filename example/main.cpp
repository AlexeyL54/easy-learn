#include "../include/ReLULayer.h"
#include "../include/SequentialModel.h"
#include "../include/SigmoidLayer.h"
#include "../include/TanhLayer.h"
#include <iostream>
#include <ostream>
#include <vector>

void sigmoidNetworkExample() {
  std::cout << "=== Сигмоидная сеть для XOR ===" << std::endl;

  {
    SequentialModel model;
    model.addLayer(std::make_unique<SigmoidLayer>(2, 4, ""));
    model.addLayer(std::make_unique<SigmoidLayer>(4, 1, ""));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    for (int epoch = 0; epoch < 1000; epoch++) {
      if (epoch % 100 == 0) {
        std::cout << "Эпоха " << epoch << ": ";
        model.train_epoch(inputs, targets, 0.5, true);
      } else {
        model.train_epoch(inputs, targets, 0.5, false);
      }
    }

    std::cout << "Результаты:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto prediction = model.predict(inputs[i]);
      std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = "
                << prediction[0] << " (ожидалось: " << targets[i][0] << ")"
                << std::endl;
    }
  }
}

void tanhNetworkExample() {
  std::cout << "\n=== Tanh сеть для XOR ===" << std::endl;

  {
    SequentialModel model;
    model.addLayer(std::make_unique<TanhLayer>(2, 4, ""));
    model.addLayer(std::make_unique<TanhLayer>(4, 1, ""));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {
        {-1}, {1}, {1}, {-1}}; // tanh выход в диапазоне [-1,1]

    for (int epoch = 0; epoch < 1000; epoch++) {
      if (epoch % 100 == 0) {
        std::cout << "Эпоха " << epoch << ": ";
        model.train_epoch(inputs, targets, 0.1, true);
      } else {
        model.train_epoch(inputs, targets, 0.1, false);
      }
    }

    std::cout << "Результаты:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto prediction = model.predict(inputs[i]);
      std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = "
                << prediction[0] << " (ожидалось: " << targets[i][0] << ")"
                << std::endl;
    }
  }
}

void mixedNetworkExample() {
  std::cout << "\n=== Смешанная сеть для XOR ===" << std::endl;

  {
    SequentialModel model;
    model.addLayer(std::make_unique<ReLULayer>(2, 4, ""));
    model.addLayer(std::make_unique<SigmoidLayer>(4, 1, ""));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    for (int epoch = 0; epoch < 1000; epoch++) {
      if (epoch % 100 == 0) {
        std::cout << "Эпоха " << epoch << ": ";
        model.train_epoch(inputs, targets, 0.1, true);
      } else {
        model.train_epoch(inputs, targets, 0.1, false);
      }
    }

    std::cout << "Результаты:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto prediction = model.predict(inputs[i]);
      std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = "
                << prediction[0] << " (ожидалось: " << targets[i][0] << ")"
                << std::endl;
    }
  }
}

int main() {
  // Тест XOR с разными архитектурами
  sigmoidNetworkExample();
  tanhNetworkExample();
  mixedNetworkExample();

  return 0;
}
