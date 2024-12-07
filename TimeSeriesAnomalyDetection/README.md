
# GRU-based Anomaly Detection Model

## Overview

This project focuses on building a **GRU-based anomaly detection model** for analyzing valve flow experiment data. The goal is to detect anomalies in the flow readings of valves using a time-series dataset. The model leverages **Gated Recurrent Units (GRU)** for sequence modeling, and its performance is evaluated using **reconstruction error** and **F1 score**.

## Data Source
Iurii D. Katser and Vyacheslav O. Kozitsin, “Skoltech Anomaly Benchmark (SKAB).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1693952.

## Model Architecture

The model architecture is based on **GRU (Gated Recurrent Units)**, a type of recurrent neural network (RNN) particularly well-suited for modeling sequential data. GRUs are effective at capturing long-range dependencies in time-series data, making them ideal for detecting anomalies in valve flow experiments.

### Key Design Decisions:

- **GRU-based Model**: The model uses GRU layers to process the time-series data and identify patterns over time. 
- **Training Setup**:
  - **Learning Rate**: 0.0005 with **AdamW** optimizer and weight decay of 0.001.
  - **Dropout**: Increased from 0.2 to 0.4 to prevent overfitting and improve model generalization.
  - **Epochs**: Trained for a total of 25 epochs.
  - **Batch Size**: Increased to 64 for better performance vs 32.
  - **Scheduler**: A learning rate scheduler that reduces the learning rate by 0.5 if the validation loss does not improve after 4 epochs.

- **Class Imbalance Handling**: Used a WeightedRandomSampler and CrossEntropyLoss to penalize misclassifications in the minority class more heavily.

### Performance:

- **Test Loss**: 0.5344
- **Test Accuracy**: 72.49%
- **Test F1 Score**: 0.7386

## Workflow

1. **Data Preprocessing**:
   - The dataset consists of time-series data organized into multiple subdirectories (`anomaly-free`, `other`, `valve1`, `valve2`).
   - The data is preprocessed by normalizing the values and preparing sequences for input to the GRU model.
   - The data is then split into training/validation/test.

2. **Training**:
   - The GRU-based model is trained using sequences 8 rows long.
   - Hyperparameters are tuned using the training/validation data.

3. **Evaluation**:
   - The model is evaluated using the **validation loss** and **F1 score**. The F1 score is used as a secondary metric to assess the model's ability to detect both anomalies and normal data.

4. **Saving the Best Model**:
   - The best-performing model based on validation loss is saved during training and used for evaluation on the test data.

## Results

After 25 epochs of training, the model achieved the following results on the test data:

- **Test Loss**: 0.5344
- **Test Accuracy**: 72.49%
- **Test F1 Score**: 0.7386

## Future Improvements

- **Hyperparameter Tuning**: Experimenting with different GRU configurations, batch sizes, and learning rates could improve performance further.
- **Model Ensembles**: Combining multiple models (e.g., GRU and other anomaly detection models) could provide more robust results.


## Conclusion

The GRU-based anomaly detection model successfully detects anomalies in valve flow experiment data by learning normal system behavior and identifying deviations. The model provides solid performance in detecting anomalies and offers a robust starting point for future enhancements.
