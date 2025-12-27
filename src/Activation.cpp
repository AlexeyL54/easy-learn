#include "../include/Activation.h"
#include <cmath>

// Helper activation functions
namespace activation {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double sigmoid_derivative(double x) { return x * (1.0 - x); }

double tanh(double x) { return std::tanh(x); }

double tanh_derivative(double x) { return 1.0 - x * x; }
} // namespace activation
