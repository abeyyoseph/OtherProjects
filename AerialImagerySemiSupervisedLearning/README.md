# Semi-Supervised Image Classification: Flooded vs. Non-Flooded Aerial Images

This project implements a semi-supervised learning approach to classify aerial images captured by UAVs after a hurricane into two categories: **Flooded** and **Non-Flooded**. The primary challenge is effectively leveraging both labeled and unlabeled data to improve classification accuracy.

---

## Dataset

The dataset is from the Floodnet challenge (https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge):

- **Labeled Data**: Contains images with known labels for "Flooded" and "Non-Flooded."
- **Unlabeled Data**: Contains images without labels for semi-supervised training.
- **Validation Data**: Contains unlabeled images used to compute the average confidence score.
- **Test Data**: Contains unlabeled images used for model inference.

---

## Objectives

1. Build a classifier that performs well on labeled data.
2. Use pseudo-labeling to integrate unlabeled data into the training process.
3. Evaluate the model's performance using the **average confidence score** as the main metric for unlabeled validation data. Since this was a challenge,
the labels for the test data were not available.

---

## Model Architecture

The model uses a **Vision Transformer (ViT)** as the backbone with transfer learning. The final classification layer is replaced with a fully connected layer and fine-tuned for the specific task.

---

## Training Strategy

### Initial Training
- The model is trained on the labeled data with the following setup:
  - **Optimizer**: AdamW
  - **Loss Function**: CrossEntropyLoss
  - **Scheduler**: ReduceLROnPlateau (decreases learning rate if loss/validation confidence does not improve for 3 epochs).
  - **Data Augmentation**: Horizontal Flip, Color Jitter, Random Rotation, Random Erasing, Gaussian Blur, Normalize.
  - **Class Imbalance**: Due to large class imbalance in the labeled training data, a `WeightedRandomSampler` was used to ensure balanced representation of labeled and pseudo-labeled samples in each batch.

### Retraining with Pseudo-Labels
1. Generate pseudo-labels for the unlabeled dataset using the initial model.
2. Combine labeled data and pseudo-labeled data into an augmented dataset.
3. Retrain the model on the augmented dataset, saving the best-performing model based on validation confidence.

---

## Metrics

### Main Metric: Average Confidence Score
- For validation and test datasets (unlabeled), the average confidence score of predictions is computed:
  \[
  \text{Average Confidence} = \frac{1}{N} \sum_{i=1}^N \max(\text{Softmax}(outputs[i]))
  \]

- The best-performing model achieves an **average confidence of 0.9659** on the validation dataset after 20 epochs.

---


## Future Work

1. Experiment with alternative semi-supervised learning techniques (e.g., Mean Teacher, FixMatch).
2. Explore using a different backbone model (e.g., ResNet or EfficientNet) to compare performance.
3. Optimize the pseudo-label confidence threshold to balance between quality and quantity of pseudo-labeled data.

