# LSTM and GRU Models for Time-Series Prediction

This project implements and compares LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models for multivariate time-series prediction. The goal is to predict the next set of time-series values based on the historical sequence of data points. 

## Dataset Description

The dataset spans **1 year** with **24 data points per day**. The developed models focused on accurately predicting NO2/CO concentration and temperature, in Celsius.

### Preprocessing
- Data was scaled using Min-Max Scaling to ensure all features fall within the range [0, 1].
- The time-series data was split into **train**, **validation**, and **test** sets, maintaining the temporal order.

## Model Architectures

### LSTM Model
The LSTM model captures long-term dependencies in time-series data using memory cells and gates.

### GRU Model
The GRU model is a simpler alternative to LSTM with fewer parameters, relying on reset and update gates for sequence modeling.

### Common Features
- Input: A sequence of historical values with `sequence_length` (e.g., 168 time steps for 1 week).
- Output: The next set of predicted values (multivariate prediction for 3 features).
- Architecture:
  - Input Layer
  - Multiple Recurrent Layers (LSTM or GRU)
  - Dropout (0.2 for regularization)
  - Fully Connected Output Layer
- Activation Functions: ReLU for intermediate layers, linear for output.

## Hyperparameters
- Sequence Length: Tested multiple values, such as 24, 48, and 168.
- Hidden Sizes: [16, 32, 48]
- Number of Layers: [1, 2, 3]
- Batch Sizes: [8, 16, 32]
- Learning Rate: 0.001
- Weight Decay: 0.01 (L2 regularization)
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Regularization: Dropout (0.2)

## Training
- Training was conducted for **50 epochs** with an early stopping mechanism based on validation loss (patience = 5).
- A **ReduceLROnPlateau** scheduler was used to dynamically adjust the learning rate when validation loss plateaued.
- Best model weights were saved during training based on validation loss.

### Metrics
Three metrics were used for evaluation:
1. **Mean Squared Error (MSE)**: Measures average squared differences.
2. **Mean Absolute Error (MAE)**: Measures average absolute differences.
3. **R-squared (R²)**: Indicates the proportion of variance explained by the model.

## Results

| Model      | Hidden Size | Layers | Batch Size | MSE (Test) | MAE (Test) | R² (Test) |
|------------|-------------|--------|------------|------------|------------|-----------|
| GRU        | 16          | 1      | 32         | 0.0190     | 0.0710     | 0.7053    |
| LSTM       | 16          | 1      | 32         | 0.0215     | 0.0753     | 0.6921    |

### Observations
- The **GRU model** slightly outperformed the LSTM model in terms of MSE, MAE, and R².
- Performance improvements stabilized early in training (around 6-7 epochs), suggesting that further hyperparameter tuning or additional regularization might be beneficial.

## Key Design Decisions
1. **Sequence Length**: Based on data granularity (24 data points/day), a weekly sequence length (168) was selected to capture longer patterns.
2. **Comparison of Models**: Both GRU and LSTM were tested to assess which model handled the task better.
3. **Dynamic Learning Rate**: A scheduler adjusted the learning rate based on validation loss to ensure stable convergence.
4. **Evaluation Metrics**: Multiple metrics were employed to provide a comprehensive understanding of the model's performance.
5. **Early Stopping**: Prevented overfitting by stopping training once validation performance stopped improving.

## Future Work
- **Explore Deeper Architectures**: Investigate deeper LSTM and GRU architectures for further improvements.
- **Attention Mechanisms**: Integrate attention-based models to improve sequence-to-sequence learning.


