# easy-learn

A lightweight C++ neural network implementation built from scratch with support for multiple activation functions, modular layer architecture. This project is inspired by the book "Deep Learning from Scratch: Building with Python from First Principles" by Seth Weidman.

## 📦 Project Structure

```
.
├── example/
│   ├── main.cpp          # Example usage
│   └── Makefile          # Build configuration
├── include/
│   ├── Activation.h      # Activation function utilities
│   ├── Layer.h           # Abstract layer interface
│   ├── ReLULayer.h       # ReLU layer implementation
│   ├── SequentialModel.h # Neural network model
│   ├── SigmoidLayer.h    # Sigmoid layer implementation
│   └── TanhLayer.h       # Tanh layer implementation
├── src/
│   ├── Activation.cpp
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

### Sequential Model
The `SequentialModel` class manages a sequence of layers and provides:
- Layer addition with `addLayer()`
- Prediction with `predict()`
- Training with `train()` and `train_epoch()`
- Full model serialization

### Activation Functions

| Function | Range | Initialization | Derivative |
|----------|-------|----------------|------------|
| Sigmoid | (0, 1) | Xavier/Glorot | f(x)(1-f(x)) |
| Tanh | (-1, 1) | Xavier/Glorot | 1 - f(x)² |
| ReLU | [0, ∞) | He | 0 if x≤0, 1 if x>0 |

## 📊 XOR Problem Results

The framework successfully learns the XOR function with all three architectures:

| Architecture | Layers | Learning Rate | Target Output | Epochs |
|--------------|--------|---------------|---------------|--------|
| Sigmoid | Sigmoid(2,4) → Sigmoid(4,1) | 0.5 | [0, 1] | 1000 |
| Tanh | Tanh(2,4) → Tanh(4,1) | 0.1 | [-1, 1] | 1000 |
| Mixed | ReLU(2,4) → Sigmoid(4,1) | 0.1 | [0, 1] | 1000 |

## 🛠️ Usage Example

```cpp
#include "include/SequentialModel.h"
#include "include/SigmoidLayer.h"

int main() {
    SequentialModel model;
    
    // Add layers
    model.addLayer(std::make_unique<SigmoidLayer>(2, 4, "layer1.txt"));
    model.addLayer(std::make_unique<SigmoidLayer>(4, 1, "layer2.txt"));
    
    // Training data
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    
    // Train
    for (int epoch = 0; epoch < 1000; epoch++) {
        model.train_epoch(inputs, targets, 0.5, epoch % 100 == 0);
    }
    
    // Predict
    auto prediction = model.predict({1, 0});
    std::cout << "Prediction: " << prediction[0] << std::endl;
    
    // Save model
    model.saveParams();
    
    return 0;
}
```

## 🔧 Extending the Framework

To add a new activation function:

1. Create a new layer class inheriting from `Layer`
2. Implement the required virtual methods
3. Add appropriate weight initialization (Xavier for sigmoid/tanh, He for ReLU)
4. Implement the activation function and its derivative

## 📚 Inspiration

This project is inspired by the excellent book **"Deep Learning from Scratch: Building with Python from First Principles" by Seth Weidman**. While implemented in C++ rather than Python, it follows similar educational principles of building neural networks from the ground up to deeply understand their inner workings.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Known Issues

- Fixed-size architecture (cannot change layer sizes after construction)
- Basic error handling
- No GPU acceleration

---

*Built for learning and experimentation with neural network fundamentals.*
