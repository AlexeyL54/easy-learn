# easy-learn

A lightweight C++ neural network implementation built from scratch with support for multiple activation functions, modular layer architecture, and loss functions. This project is inspired by the book "Deep Learning from Scratch: Building with Python from First Principles" by Seth Weidman.

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
│   └── SequentialModel.h   # Neural network model
├── src/
│   ├── Activation.cpp
│   ├── MSE.cpp             # MSE loss implementation
│   ├── ReLULayer.cpp
│   ├── SequentialModel.cpp
│   ├── SigmoidLayer.cpp
│   └── TanhLayer.cpp
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
- **Sequential Model**: Simple feedforward neural network builder
- **Backpropagation**: Full backpropagation implementation with gradient descent
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
- `backward()`: Perform backpropagation and weight updates
- `getWeights()` / `setWeights()`: Access layer parameters
- `saveParams()` / `downloadParams()`: Serialize/deserialize layer state

### Loss Functions
The framework includes abstract `Loss` class with:
- `computeLoss()`: Calculate loss between prediction and target
- `computeGrad()`: Compute gradient for backpropagation
Currently implemented: **Mean Squared Error (MSE)**

### Sequential Model
The `SequentialModel` class manages a sequence of layers and provides:
- Layer addition with `addLayer()`
- Prediction with `predict()`
- Training with `train()` and `train_epoch()`
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

int main() {
    // Build layers
    std::vector<std::unique_ptr<Layer>> layers;
    layers.emplace_back(std::make_unique<SigmoidLayer>(2, 4, ""));
    layers.emplace_back(std::make_unique<SigmoidLayer>(4, 1, ""));
    
    // Create model with MSE loss
    SequentialModel model(std::move(layers), 
                         std::make_unique<MSE>(), 
                         1000,  // epochs
                         0.5);  // learning rate
    
    // XOR training data
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    
    // Train the model
    model.train(inputs, targets);
    
    // Test predictions
    for (const auto& input : inputs) {
        auto prediction = model.predict(input);
        std::cout << input[0] << " XOR " << input[1] << " = " 
                  << prediction[0] << std::endl;
    }
    
    // Save model parameters
    model.saveParams();
    
    return 0;
}
```

## 🔧 Extending the Framework

### Adding a New Activation Function
1. Create a new layer class inheriting from `Layer`
2. Implement the required virtual methods
3. Add appropriate weight initialization (Xavier for sigmoid/tanh, He for ReLU)
4. Implement the activation function and its derivative

### Adding a New Loss Function
1. Create a new class inheriting from `Loss`
2. Implement `computeLoss()` and `computeGrad()` methods
3. Integrate with `SequentialModel` constructor

## 📚 Inspiration

This project is inspired by the excellent book **"Deep Learning from Scratch: Building with Python from First Principles" by Seth Weidman**. While implemented in C++ rather than Python, it follows similar principles of building neural networks from the ground up to deeply understand their inner workings.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚡In progress

- Implementation of multiple algorithms for optimization 
- Better error handling
- Soft-size architecture
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
