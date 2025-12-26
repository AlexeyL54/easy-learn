#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

using std::vector;

/*
 * @brief Шаблон слоя, содержащий самые главные действия слоя, которые должные
 * быть переопределены конкретной реализацией.
 */
class Layer {

public:
  virtual ~Layer() = default;

  /*
   * @brief Осуществить прямой проход
   * @param input выходные данные (сигналы аксонов) предыдущих нейронов
   * @return выходные данные этого слоя
   */
  virtual vector<double> forward(const vector<double> &input) = 0;

  /*
   * @brief Осуществить обратный проход (исправить веса)
   * @param output_grads градиенты предыдущих слоев
   * @param learning_rate скорость обучения
   * @return градиент
   */
  virtual vector<double> backward(const vector<double> &output_grads,
                                  double learning_rate) = 0;

  /*
   * @brief Получить значения весов в слое
   * @return веса
   */
  virtual vector<vector<double>> getWeights() const = 0;

  /*
   * @brief Задать весам новое значение
   */
  virtual void setWeights(const vector<vector<double>> &new_weights) = 0;

  /*
   * @brief Получить количество входящих связей
   * @return количество входящий связей
   */
  virtual int getInputSize() const = 0;

  /*
   * @brief Получить количество исходящий связей
   * @return количество исходящих связей
   */
  virtual int getOutputSize() const = 0;

  virtual void saveParams() = 0;

  virtual void downloadParams() = 0;
};

#endif
