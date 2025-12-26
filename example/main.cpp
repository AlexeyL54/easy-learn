#include "../LSTMLayer.hpp"
#include "../ReLULayer.hpp"
#include "../SequentualModel.hpp"
#include "../SigmoidLayer.hpp"
#include "../TanhLayer.hpp"
#include <iostream>
#include <ostream>
#include <vector>

int main() {
  // Тест XOR с разными архитектурами

  std::cout << "=== Тест 1: Сигмоидная сеть для XOR ===" << std::endl;
  {
    SequentialModel model;
    model.addLayer(std::make_unique<SigmoidLayer>(2, 4));
    model.addLayer(std::make_unique<SigmoidLayer>(4, 1));

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

  std::cout << "\n=== Тест 2: ReLU сеть для XOR ===" << std::endl;
  {
    SequentialModel model;
    model.addLayer(std::make_unique<ReLULayer>(2, 4));
    model.addLayer(std::make_unique<SigmoidLayer>(4, 1));

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

  std::cout << "\n=== Тест 3: Tanh сеть для XOR ===" << std::endl;
  {
    SequentialModel model;
    model.addLayer(std::make_unique<TanhLayer>(2, 4));
    model.addLayer(std::make_unique<TanhLayer>(4, 1));

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

  std::cout << "=== Тест 4: LSTM слоя ===" << std::endl;

  // Параметры
  int sequence_length = 5;
  int input_size = 3;
  int hidden_size = 8;
  int output_size = 2;

  // Создаем модель с LSTM слоем
  SequentialModel model;
  model.addLayer(std::make_unique<LSTMLayer>(input_size, hidden_size,
                                             output_size, sequence_length));

  // Тестовые данные: случайная последовательность
  std::vector<double> input(sequence_length * input_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < sequence_length * input_size; i++) {
    input[i] = dist(gen);
  }

  // Целевые значения
  std::vector<double> target(sequence_length * output_size);
  for (int i = 0; i < sequence_length * output_size; i++) {
    target[i] = dist(gen);
  }

  // Обучение
  std::cout << "Обучение LSTM..." << std::endl;
  for (int epoch = 0; epoch < 1000; epoch++) {
    double loss = model.train(input, target, 0.1);
    if (epoch % 100 == 0) {
      std::cout << "Эпоха " << epoch << ", Ошибка: " << loss << std::endl;
    }
  }

  // Предсказание
  auto prediction = model.predict(input);
  std::cout << "Предсказание: ";
  for (double num : prediction)
    std::cout << num;
  std::cout << std::endl;

  std::cout << "Размер предсказания: " << prediction.size() << std::endl;

  return 0;
}
