# GI Imaging Classification using Transfer Learning

## Project Overview
This project focuses on classifying gastrointestinal (GI) images using transfer learning with Vision Transformers (ViT-16) and ConvNeXt models. The dataset used is the HyperKvasir dataset (https://datasets.simula.no/hyper-kvasir/), which consists of labeled GI tract images. As with many medical datasets, there was significant class imbalance prompting the use of several techniques for improving model performance and generalization.

## Dataset
The dataset used for this project is the **HyperKvasir** dataset, which contains 23 categories of GI tract images such as Barrett's Esophagus, Polyps, Esophagitis, and Ulcerative Colitis, among others.

## Data Preprocessing & Augmentation
A **stratified split** was performed to divide the dataset into training, validation, and test sets while maintaining the class distribution.

To enhance model robustness, the following data augmentation techniques were applied to the training data:
- **Random Rotation**
- **Horizontal Flip**
- **Color Jitter**
- **Gaussian Blur**


## Class Imbalance Handling
To address class imbalance, the following techniques were implemented:
- **Weighted Cross Entropy Loss Function**: Assigning higher weights to minority classes.
- **Oversampling**: Using `WeightedRandomSampler` to ensure underrepresented classes were adequately sampled.

## Model Selection & Training
Two models were trained using transfer learning:
- **Vision Transformer (ViT-16)**
- **ConvNeXt**

Each model was initialized with pre-trained ImageNet weights and fine-tuned on the HyperKvasir dataset.

### Hyperparameters
- **Batch Size**: 64
- **Gradient Accumulation Step Size**: 2 (making equivalent batch size 128)
- **Optimizer**: AdamW
- **Initial Learning Rate**: 0.001
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Epochs**: 20
- **Weight Decay**: 0.001
- **Loss Function**: Cross-Entropy with Class Weights

## Results
### Individual Model Performance
#### ViT-16:
- **Best Validation F1 Score**: 0.8340 

#### ConvNeXt:
- **Best Validation F1 Score**: 0.7972

### Ensemble Performance
The ensemble of both models was evaluated on the test set, yielding the following metrics:
- **Test Accuracy**: 0.8239
- **Test Precision**: 0.8721
- **Test Recall**: 0.8239
- **Test F1 Score**: 0.8357

## Conclusion
This project successfully applied transfer learning to classify GI tract images. Despite the class imbalance, the use of weighted loss functions and oversampling techniques helped improve model performance. The ensemble approach further enhanced classification results, making this a viable method for automated GI image analysis.

---
**Future Work:**
- Experiment with additional augmentation techniques
- Fine-tune hyperparameters further for improved generalization
- Train the individual models for a longer time

