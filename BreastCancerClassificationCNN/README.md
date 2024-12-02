# Breast Cancer Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify histopathology images (from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) as either **cancerous** or **benign**. The dataset consists of small, color images of size 50x50 pixels. The model is designed to address class imbalance, optimize performance, and avoid overfitting while maintaining computational feasibility.

## Dataset and Preprocessing
The dataset was split into three sets using a **stratified split** to ensure equal class distribution:
- **Training Set**: 70% of the data
- **Validation Set**: 20% of the data
- **Test Set**: 10% of the data

To further address class imbalance, a **WeightedRandomSampler** was used in the training set to ensure that minority class samples were not underrepresented during training. Additionally, **Weighted CrossEntropy Loss** was used to provide class-specific penalties during backpropagation.

## Architecture
The CNN is composed of three convolutional blocks followed by fully connected layers. Each convolutional block consists of:
1. **Convolution**: Extracts spatial features using 3x3 kernels.
2. **Batch Normalization**: Stabilizes training by normalizing feature maps.
3. **Leaky ReLU Activation**: Introduced to handle vanishing gradients while allowing a small gradient for negative values.
4. **Pooling**: Reduces spatial dimensions using max pooling with a stride of 2.
5. **Dropout**: Reduces overfitting by randomly zeroing a fraction of activations.

The final fully connected layers use dropout to further regularize the model.


## Training Results
### Metrics
| Metric          | Value    |
|------------------|----------|
| Test Loss        | 0.5387   |
| Accuracy         | 78.34%   |
| Precision        | 0.8453   |
| Recall           | 0.7834   |
| F1 Score         | 0.7937   |

---

## Future Improvements
- Experiment with other activation functions such as **ELU** or **Mish** for better gradient flow.
- Fine-tune the learning rate scheduler to adapt more dynamically.
---

