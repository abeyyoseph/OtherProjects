# Breast Cancer Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for binary classification of breast histopathology images (from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) as either **cancerous** or **benign**. The dataset consists of small, color images of size 50x50 pixels. The model is designed to address class imbalance, optimize performance, and avoid overfitting while maintaining computational feasibility.

## Dataset and Preprocessing
The dataset consists of labeled medical images for two classes:
- `Benign`
- `Cancer`
The dataset was split into three sets using a **stratified split** to ensure equal class distribution:
- **Training Set**: 70% of the data
- **Validation Set**: 20% of the data
- **Test Set**: 10% of the data

## Data Augmentation
To improve generalization, the following augmentations were applied during training:
- Random horizontal and vertical flips
- Random rotations
- Random resized cropping
- Normalization using mean and std of the dataset

These augmentations helped increase robustness and mitigate overfitting.

## Architecture
The CNN is composed of three convolutional blocks followed by fully connected layers. Each convolutional block consists of:
1. **Convolution**: Extracts spatial features using 3x3 kernels.
2. **Batch Normalization**: Stabilizes training by normalizing feature maps.
3. **Leaky ReLU Activation**: Introduced to handle vanishing gradients while allowing a small gradient for negative values.
4. **Pooling**: Reduces spatial dimensions using max pooling with a stride of 2.
5. **Dropout**: Reduces overfitting by randomly zeroing a fraction of activations.

The final fully connected layers use dropout to further regularize the model.

## ⚙️ Hyperparameters
- Optimizer: AdamW
- Learning Rate: Warmup from `0.000333` → `0.001`, plateau reduction on stagnation
- Epochs: 10
- Batch Size: 32
- Loss Function: Weighted Cross Entropy
- Scheduler: Warmup followed by ReduceLROnPlateau

To further address class imbalance, a **WeightedRandomSampler** was used in the training set to ensure that minority class samples were not underrepresented during training. Additionally, **Weighted CrossEntropy Loss** was used to provide class-specific penalties during backpropagation.

## Training Results
### Metrics
| Metric          | Value    |
|------------------|----------|
| Test Loss        | 0.5387   |
| Accuracy         | 78.34%   |
| Precision        | 0.8453   |
| Recall           | 0.7834   |
| F1 Score         | 0.7937   |

![TrainingResults](InitialTraining.png)

## Test Results
- **Accuracy**: 75.65%
- **Precision**: 0.7484
- **Recall**: 0.8050
- **F1 Score**: 0.7422

### 📊 Per-Class Metrics
#### Benign
- Precision: 0.9547
- Recall: 0.6928
- F1 Score: 0.8030

#### Cancer
- Precision: 0.5420
- Recall: 0.9171
- F1 Score: 0.6814

## Summary
The CNN achieved good general performance, with better recall on cancer and higher precision on benign. Weighted loss and data augmentation were crucial. Future improvements may include threshold tuning, focal loss, or fine-tuning the learning rate scheduler to adapt more dynamically.


