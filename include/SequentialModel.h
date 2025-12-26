#pragma once

#include "Layer.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

using std::vector;

/*
 * @brief Реализовать сборку модели из слоев, обучение и предсказание
 */
class SequentialModel {
private:
  vector<std::unique_ptr<Layer>> layers;

public:
  /*
   * @brief Добавить слой в модель
   * @param layer указатель на объкт слоя
   */
  void addLayer(std::unique_ptr<Layer> layer);

  /*
   * @brief Получить выходное значение модели (предсказание)
   * @param input входные данные (признаки)
   * @return выходное значение
   */
  vector<double> predict(const vector<double> &input);

  /*
   * @brief Обучить модель
   * @param input входные данные (признаки)
   * @param target образцовые выходные данные
   * @param learning_rate скорость обученя
   * @return ошибка
   */
  double train(const vector<double> &input, const vector<double> &target,
               const double learning_rate);

  /*
   * @brief Пройти одну эпоху обучения
   * @param inputs входные данные (признаки)
   * @param targets эталонные значения выходных данных
   * @param learning_rate скорость обучения
   * @param verbose флаг вывода информации об ошибке
   */
  void train_epoch(const vector<vector<double>> &inputs,
                   const vector<vector<double>> &targets, double learning_rate,
                   bool verbose = false);

  void saveParams();

  void downloadParams();
};
