#ifndef MSE_H
#define MSE_H

#include "Loss.h"
#include <vector>

using std::vector;

/*
 * @brief Implementation of Mean Squared Error loss function
 */
class MSE : public Loss {

private:
  vector<double> prediction;
  vector<double> target;

public:
  /*
   * @brief Check shapes and compute loss
   * @param prediction - output value of model
   * @param target - target value for output
   */
  double computeLoss(vector<double> &prediction,
                     const vector<double> &target) override;

  /*
   * @brief Compute gradient in respect to loss function input values
   */
  vector<double> computeGrad() override;
};

#endif // !MSE_H
