#ifndef SGD_H
#define SGD_H

#include "../layers/Layer.h"
#include "Optimizer.h"
#include <memory>

class SGD : public Optimizer {
private:
  float learning_rate;

public:
  SGD(float lr);

  /*
   * @brief Correct weights
   * @param layer pointer to a layer object
   */
  void step(Layer &layer) override;
};

#endif // !SGD_H
