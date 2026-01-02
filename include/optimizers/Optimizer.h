#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../layers/Layer.h"
#include <memory>

/*
 * @brief Implementation of a template for optimization functions
 */
class Optimizer {
public:
  ~Optimizer() = default;

  /*
   * @brief Correct weights
   * @param layer pointer to a layer object
   */
  virtual void step(Layer &layer) = 0;
};

#endif // !OPTIMIZER_H
