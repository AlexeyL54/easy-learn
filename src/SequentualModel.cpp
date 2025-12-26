#pragma once

#include "../include/SequentialModel.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

using std::vector;

/*
 * @brief Добавить слой в модель
 * @param layer указатель на объкт слоя
 */
void SequentialModel::addLayer(std::unique_ptr<Layer> layer) {
  layers.push_back(std::move(layer));
}

/*
 * @brief Получить выходное значение модели (предсказание)
 * @param input входные данные (признаки)
 * @return выходное значение
 */
vector<double> SequentialModel::predict(const vector<double> &input) {
  vector<double> activation = input;

  for (std::unique_ptr<Layer> &layer : layers) {
    activation = layer->forward(activation);
  }
  return activation;
}

/*
 * @brief Обучить модель
 * @param input входные данные (признаки)
 * @param target образцовые выходные данные
 * @param learning_rate скорость обученя
 * @return ошибка
 */
double SequentialModel::train(const vector<double> &input,
                              const vector<double> &target,
                              const double learning_rate) {
  double loss = 0.0;
  double error = 0.0;
  vector<double> output = predict(input);
  vector<double> gradient(output.size());

  // Вычислить ошибку MSE
  for (size_t i = 0; i < output.size(); i++) {
    error = output[i] - target[i];
    loss += error * error;
  }
  loss /= output.size();

  // Вычислить градиенты
  for (size_t i = 0; i < output.size(); i++) {
    gradient[i] = 2.0 * (output[i] - target[i]) / output.size();
  }

  // Обратный проход и корректировка весов
  for (std::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it) {
    gradient = (*it)->backward(gradient, learning_rate);
  }
  return loss;
}

/*
 * @brief Пройти одну эпоху обучения
 * @param inputs входные данные (признаки)
 * @param targets эталонные значения выходных данных
 * @param learning_rate скорость обучения
 * @param verbose флаг вывода информации об ошибке
 */
void SequentialModel::train_epoch(const vector<vector<double>> &inputs,
                                  const vector<vector<double>> &targets,
                                  double learning_rate, bool verbose) {
  double total_loss = 0.0;

  for (size_t i = 0; i < inputs.size(); i++) {
    total_loss += train(inputs[i], targets[i], learning_rate);
  }

  if (verbose) {
    std::cout << "Средняя ошибка за эпоху: " << total_loss / inputs.size()
              << std::endl;
  }
}

void SequentialModel::saveParams() {
  for (std::unique_ptr<Layer> &layer : layers) {
    layer->saveParams();
  }
}

void SequentialModel::downloadParams() {
  for (std::unique_ptr<Layer> &layer : layers) {
    layer->downloadParams();
  }
};
