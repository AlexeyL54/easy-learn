#ifndef LOSS_H
#define LOSS_H

#include <vector>

using std::vector;

class Loss {

public:
  virtual ~Loss() = default;

  /*
   * @brief Check shapes and compute loss
   * @param prediction - output value of model
   * @param target - target value for output
   */
  virtual double computeLoss(vector<double> &prediction,
                             const vector<double> &target) = 0;

  /*
   * @brief Compute gradient in respect to loss function input values
   */
  virtual vector<double> computeGrad() = 0;
};

#endif
