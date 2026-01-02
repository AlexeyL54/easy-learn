# easy-learn

A lightweight C++ neural network implementation built from scratch with support for multiple activation functions, modular layer architecture, loss functions, and optimizers. This project is inspired by the book "Deep Learning from Scratch: Building with Python from First Principles" by Seth Weidman.

## 📦 Project Structure

```.
├── example/
│   ├── main.cpp            # Example usage
│   └── Makefile            # Build configuration
├── include/
│   ├── Activation.h        # Activation function utilities
│   ├── layers/
│   │   ├── Layer.h         # Abstract layer interface
│   │   ├── ReLULayer.h     # ReLU layer implementation
│   │   ├── SigmoidLayer.h  # Sigmoid layer implementation
│   │   └── TanhLayer.h     # Tanh layer implementation
│   ├── loss/
│   │   ├── Loss.h          # Abstract loss interface
│   │   └── MSE.h           # Mean Squared Error implementation
│   ├── optimizers/
│   │   ├── Optimizer.h     # Abstract optimizer interface
│   │   └── SGD.h           # Stochastic Gradient Descent implementation
│   └── SequentialModel.h   # Neural network model
├── src/
│   ├── Activation.cpp
│   ├── layers/
│   │   ├── ReLULayer.cpp
│   │   ├── SigmoidLayer.cpp
│   │   └── TanhLayer.cpp
│   ├── loss/
│   │   └── MSE.cpp
│   ├── optimizers/
│   │   └── SGD.cpp
│   └── SequentialModel.cpp
├── LICENSE
└── README.md
```

## ✨ Features

- **Modular Layer Architecture**: Easily extendable layer system with abstract base class
- **Multiple Activation Functions**:
  - Sigmoid with Xavier/Glorot initialization
  - Tanh with Xavier/Glorot initialization  
  - ReLU with He initialization
- **Loss Functions**: Mean Squared Error (MSE) implementation
- **Optimizers**: Stochastic Gradient Descent (SGD)
- **Sequential Model**: Simple feedforward neural network builder with integrated training loop
- **Backpropagation**: Full backpropagation implementation with separated gradient computation and weight update steps
- **Model Persistence**: Save and load layer weights and biases to/from files
- **XOR Problem Demo**: Ready-to-run examples demonstrating different architectures

## 🚀 Getting Started

### Prerequisites
- C++ compiler with C++11 support (g++ recommended)
- Make build system

### Building the Example
```bash
cd example
make
./example.out 
```

## 🧠 Architecture

### Layer Interface
All layers implement the abstract `Layer` class with these key methods:
- `forward()`: Perform forward propagation
- `backward()`: Perform backpropagation (gradient computation only)
- `getWeights()` / `setWeights()`: Access layer parameters
- `getWeightGrads()` / `getBiasGrads()`: Access computed gradients
- `saveParams()` / `downloadParams()`: Serialize/deserialize layer state

### Loss Functions
The framework includes abstract `Loss` class with:
- `computeLoss()`: Calculate loss between prediction and target
- `computeGrad()`: Compute gradient for backpropagation
Currently implemented: **Mean Squared Error (MSE)**

### Optimizers
The `Optimizer` abstract class defines the interface for weight update algorithms:
- `step()`: Update layer parameters using computed gradients
Currently implemented: **Stochastic Gradient Descent (SGD)**

### Sequential Model
The `SequentialModel` class manages a sequence of layers and provides:
- Prediction with `predict()`
- Training loop with `train()`
- Backward pass coordination with `backward()`
- Full model serialization

### Activation Functions

| Function | Range | Initialization | Derivative | Use Case |
|----------|-------|----------------|------------|----------|
| Sigmoid | (0, 1) | Xavier/Glorot | f(x)(1-f(x)) | Binary classification, output layer |
| Tanh | (-1, 1) | Xavier/Glorot | 1 - f(x)² | Hidden layers, regression |
| ReLU | [0, ∞) | He | 0 if x≤0, 1 if x>0 | Hidden layers, deep networks |

## 🛠️ Usage Example

```cpp
#include "include/SequentialModel.h"
#include "include/layers/SigmoidLayer.h"
#include "include/loss/MSE.h"
#include "include/optimizers/SGD.h"
#include <memory>
#include <vector>

std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

int main() {
  std::cout << "=== Sigmoid net for XOR ===" << std::endl;

  // build layers
  std::vector<std::unique_ptr<Layer>> layers;
  layers.emplace_back(std::make_unique<ReLULayer>(2, 8, "layer1.txt"));
  layers.emplace_back(std::make_unique<TanhLayer>(8, 4, "layer2.txt"));
  layers.emplace_back(std::make_unique<SigmoidLayer>(4, 1, "layer3.txt"));

  // create model
  SequentialModel model(std::move(layers), std::make_unique<MSE>(),
                        std::make_unique<SGD>(0.1), 1000);

  // train model
  model.train(inputs, targets);

  std::cout << "Results:" << std::endl;
  for (size_t i = 0; i < inputs.size(); i++) {
    vector<double> prediction = model.predict(inputs[i]);
    std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = "
              << prediction[0] << " (expected: " << targets[i][0] << ")"
              << std::endl;
  }
    
  return 0;
}
```

## 🔧 Extending the Framework

### Adding a New Activation Function
1. Create a new layer class inheriting from `Layer`
2. Implement the required virtual methods
3. Add appropriate weight initialization (Xavier for sigmoid/tanh, He for ReLU)
4. Implement the activation function and its derivative in backward pass

### Adding a New Loss Function
1. Create a new class inheriting from `Loss`
2. Implement `computeLoss()` and `computeGrad()` methods
3. Integrate with `SequentialModel` constructor

### Adding a New Optimizer
1. Create a new class inheriting from `Optimizer`
2. Implement `step()` method to update weights using gradients
3. Use with `SequentialModel` for training

## 📚 Implementation Details

The framework implements a clear separation of concerns:
1. **Forward pass**: Layers compute activations and cache intermediate values
2. **Loss computation**: Loss function calculates error and gradient
3. **Backward pass**: Layers compute gradients (stored in `*_grads` members)
4. **Optimization**: Optimizer updates weights using computed gradients

This separation allows for flexible optimizer implementations and easy debugging.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚡ In Progress

- Implementation of multiple optimization algorithms (Adam, RMSprop)
- Better error handling and validation
- More layer types (Dropout, BatchNorm)
- Multi-thread architecture
- GPU acceleration
- Saving model to ONNX format

## 🐛 Limitations

- Fixed-size architecture (cannot change layer sizes after construction)
- Basic error handling
- No GPU acceleration
- Limited to fully connected layers
- Single-threaded implementation

---
