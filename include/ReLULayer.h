#pragma once

#include "Layer.h"
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

/*
 * @brief Реализация слоя с функцией активации ReLU
 */
class ReLULayer : public Layer {
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
  ReLULayer(int input, int neurons, std::string file_name);

  void saveParams() override;

  void downloadParams() override;

  /*
   * @brief Осуществить прямой проход
   * @param input выходные данные (сигналы аксонов) предыдущих нейронов
   * @return выходные данные этого слоя
   */
  vector<double> forward(const std::vector<double> &input) override;

  /*
   * @brief Осуществить обратный проход (исправить веса)
   * @param output_grads градиенты предыдущих слоев
   * @param learning_rate скорость обучения
   * @return градиент
   */
  vector<double> backward(const std::vector<double> &output_gradient,
                          double learning_rate) override;

  /*
   * @brief Получить значения весов в слое
   * @return веса
   */
  vector<vector<double>> getWeights() const override;

  /*
   * @brief Задать весам новое значение
   */
  void setWeights(const vector<vector<double>> &new_weights) override;

  /*
   * @brief Получить количество входящих связей
   * @return количество входящий связей
   */
  int getInputSize() const override;

  /*
   * @brief Получить количество исходящий связей
   * @return количество исходящих связей
   */
  int getOutputSize() const override;
};
