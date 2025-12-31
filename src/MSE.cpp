#include "../include/loss/MSE.h"
#include <vector>

using std::vector;

/*
 * @brief Check shapes and compute loss
 * @param prediction - output value of model
 * @param target - target value for output
 */
double MSE::computeLoss(vector<double> &prediction,
                        const vector<double> &target) {
  double error = 0;
  double loss = 0;

  this->prediction = prediction;
  this->target = target;

  for (size_t i = 0; i < prediction.size(); i++) {
    error = prediction[i] - target[i];
    loss += error * error;
  }

  return loss / prediction.size();
}

/*
 * @brief Compute gradient in respect to loss function input values
 */
vector<double> MSE::computeGrad() {
  vector<double> gradient(prediction.size());

  for (size_t i = 0; i < prediction.size(); i++) {
    gradient[i] = 2.0 * (prediction[i] - target[i]) / prediction.size();
  }

  return gradient;
};
