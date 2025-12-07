# 🔬 Vision Transformers & CNN Laboratory

> *An end-to-end deep learning journey through modern computer vision architectures*

This comprehensive PyTorch-powered laboratory provides hands-on exploration of cutting-edge computer vision techniques, featuring detailed comparative analysis of multiple neural network architectures on the MNIST dataset.

---

## 🎯 Project Objectives

This laboratory is designed to build deep, practical expertise in modern computer vision through systematic implementation, training, and evaluation of diverse deep learning models:

- **Architecture Design** — Crafting neural networks from the ground up using PyTorch's flexible framework
- **Multi-Model Implementation** — Building and comparing CNNs, Faster R-CNN, Fully Connected Networks, and Vision Transformers
- **Transfer Learning Mastery** — Leveraging powerful pre-trained models including VGG16 and AlexNet
- **Performance Optimization** — GPU acceleration, hyperparameter tuning, and comprehensive performance analysis
- **Architectural Trade-offs** — Understanding the balance between lightweight custom models and heavyweight pre-trained architectures

---

## 🧠 Part 1: Convolutional Neural Networks & Object Detection

### Implementation Highlights

**Custom CNN Architecture**
- Multi-layer convolutional networks with pooling and fully connected layers
- Configurable kernel sizes, padding strategies, and stride patterns
- Advanced activation functions and regularization techniques
- Optimizer selection and learning rate strategies

**Faster R-CNN for Digit Detection**
- Region proposal networks for object localization
- End-to-end digit detection and classification
- Real-time inference capabilities

**GPU-Accelerated Training**
- CUDA-optimized training pipelines
- Efficient batch processing and memory management

### Comprehensive Evaluation Metrics

- **Accuracy** — Overall classification performance
- **F1-Score** — Balanced precision-recall analysis
- **Loss Curves** — Training dynamics and convergence behavior
- **Training Time** — Computational efficiency benchmarks

### Transfer Learning Experiments

Deep comparison using industry-standard pre-trained models:
- **VGG16** — Deep architecture with small receptive fields
- **AlexNet** — Pioneering deep CNN architecture

### Model Comparison Matrix

Final performance analysis across all implemented architectures:
- Custom CNN baseline
- Faster R-CNN object detection
- VGG16 transfer learning
- AlexNet transfer learning

---

## ⚡ Part 2: Vision Transformer (ViT) from Scratch

A complete implementation of the Vision Transformer architecture, following the groundbreaking work by Dosovitskiy et al. (2020), demonstrating the power of attention mechanisms in computer vision.

### Core Components Implemented

**Patch Embedding Layer**
- Image tokenization through patch extraction
- Linear projection to transformer dimension

**Multi-Head Self-Attention**
- Parallel attention mechanisms
- Query-Key-Value transformations
- Attention weight visualization

**Positional Encoding**
- Learnable position embeddings
- Spatial relationship preservation

**Transformer Encoder Blocks**
- Layer normalization
- Feed-forward networks
- Residual connections

**Classification Head**
- Global pooling strategies
- Final prediction layer

### Evaluation & Benchmarking

- Performance metrics on MNIST dataset
- Direct comparison with CNN and Faster R-CNN architectures
- Analysis of attention patterns and learned representations

---

## 🚀 Key Takeaways

This laboratory provides invaluable insights into:
- The evolution from convolutional to attention-based architectures
- Practical considerations in model selection and deployment
- The role of transfer learning in accelerating development
- Trade-offs between model complexity, accuracy, and efficiency

---

## 📊 Technologies & Frameworks

- **PyTorch** — Deep learning framework
- **CUDA** — GPU acceleration
- **NumPy** — Numerical computing
- **Matplotlib** — Visualization
- **MNIST** — Benchmark dataset

---

*Built with passion for advancing computer vision understanding* 🎓
