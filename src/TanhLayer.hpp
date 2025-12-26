#pragma once

#include "Layer.hpp"
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

/*
 * @brief Реализация слоя с функцией активации: гиперболический тангенс
 */
class TanhLayer : public Layer {
private:
  vector<vector<double>> weights; // веса входов для каждого нейрона слоя
  vector<double> biases;          // смещения для каждого нейрона
  vector<double> last_input;      // последние входные данные
  vector<double> last_output;     // последние выходные данные
  vector<double> last_z;          // взвешенная сумма
  int input_size;
  int output_size;
  std::string config_name;

public:
  TanhLayer(int input, int neurons, std::string file_name) {
    input_size = input;
    output_size = neurons;
    config_name = file_name;
    // Инициализация весов методом Xavier/Glorot
    std::random_device rd;
    std::mt19937 gen(rd());
    double stddev = std::sqrt(2.0 / (input_size + output_size));
    std::normal_distribution<double> dist(0.0, stddev);

    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size, 0.1);

    for (int i = 0; i < output_size; i++) {
      for (int j = 0; j < input_size; j++) {
        weights[i][j] = dist(gen);
      }
    }
  }

  void saveParams() override {
    std::ofstream file(config_name);

    if (file.is_open()) {

      file << input_size << "\n";
      file << output_size << "\n";

      for (vector<double> &vec : weights) {
        for (double weight : vec) {
          file << weight << " ";
        }
        file << "\n";
      }
      for (double &bias : biases) {
        file << bias << " ";
      }
      file.close();
    }
  }

  void downloadParams() override {
    std::string line;
    double value;
    std::ifstream file(config_name);

    if (file.is_open()) {
      std::getline(file, line);
      input_size = std::stoi(line);

      std::getline(file, line);
      output_size = std::stoi(line);

      // ИНИЦИАЛИЗИРУЕМ ВЕКТОР ПРАВИЛЬНЫМ РАЗМЕРОМ ПЕРЕД ЧТЕНИЕМ
      weights.resize(output_size, std::vector<double>(input_size));

      // Чтение весов
      for (int i = 0; i < output_size; i++) {
        std::getline(file, line);
        std::stringstream s(line);
        vector<double> row_weights;

        while (s >> value) {
          row_weights.push_back(value);
        }

        // ПРОВЕРКА РАЗМЕРА
        if (row_weights.size() != static_cast<size_t>(input_size)) {
          throw std::runtime_error("Weight size mismatch in TanhLayer");
        }

        weights[i] = row_weights;
      }

      // ИНИЦИАЛИЗИРУЕМ СМЕЩЕНИЯ
      biases.resize(output_size);

      // Чтение смещений
      std::getline(file, line);
      std::stringstream s(line);
      vector<double> loaded_biases;

      while (s >> value) {
        loaded_biases.push_back(value);
      }

      // ПРОВЕРКА РАЗМЕРА
      if (loaded_biases.size() != static_cast<size_t>(output_size)) {
        throw std::runtime_error("Bias size mismatch in TanhLayer");
      }

      biases = loaded_biases;
      file.close();
    }
  }

  /*
   * @brief Осуществить прямой проход
   * @param input выходные данные (сигналы аксонов) предыдущих нейронов
   * @return выходные данные этого слоя
   */
  vector<double> forward(const vector<double> &input) override {
    last_input = input;
    int output_size = weights.size();
    vector<double> output(output_size);
    last_z.resize(output_size);

    for (size_t i = 0; i < output_size; i++) {
      last_z[i] = biases[i];

      for (size_t j = 0; j < input.size(); j++) {
        last_z[i] += weights[i][j] * input[j];
      }
      // Tanh активация
      output[i] = std::tanh(last_z[i]);
    }

    last_output = output;
    return output;
  }

  /*
   * @brief Осуществить обратный проход (исправить веса)
   * @param output_grads градиенты предыдущих слоев
   * @param learning_rate скорость обучения
   * @return градиент
   */
  std::vector<double> backward(const std::vector<double> &output_gradient,
                               double learning_rate) override {
    int input_size = last_input.size();
    int output_size = weights.size();
    std::vector<double> input_gradient(input_size, 0.0);

    // Вычисляем градиент относительно взвешенной суммы (z)
    std::vector<double> z_gradient(output_size);
    for (int i = 0; i < output_size; i++) {
      double activation_derivative = 1.0 - last_output[i] * last_output[i];
      z_gradient[i] = output_gradient[i] * activation_derivative;
    }

    // Вычисляем градиенты для весов и обновляем их
    for (int i = 0; i < output_size; i++) {
      for (int j = 0; j < input_size; j++) {
        double weight_gradient = z_gradient[i] * last_input[j];
        weights[i][j] -= learning_rate * weight_gradient;

        // Накопление градиента для входного слоя
        input_gradient[j] += z_gradient[i] * weights[i][j];
      }
      biases[i] -= learning_rate * z_gradient[i];
    }

    return input_gradient;
  }

  /*
   * @brief Получить значения весов в слое
   * @return веса
   */
  vector<vector<double>> getWeights() const override { return weights; }

  /*
   * @brief Задать весам новое значение
   */
  void setWeights(const vector<vector<double>> &new_weights) override {
    weights = new_weights;
  }

  /*
   * @brief Получить количество входящих связей
   * @return количество входящий связей
   */
  int getInputSize() const override { return weights[0].size(); }

  /*
   * @brief Получить количество исходящий связей
   * @return количество исходящих связей
   */
  int getOutputSize() const override { return weights.size(); }
};
