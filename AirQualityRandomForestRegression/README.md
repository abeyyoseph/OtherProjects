# Random Forest Regression Model for NO₂ Concentration Prediction

## Project Overview

This project aims to predict Nitrogen Dioxide (NO₂) concentrations using a Random Forest Regression model. The dataset contains time-series air quality data collected over one year, with 24 data points per day (from https://www.kaggle.com/datasets/dakshbhalala/uci-air-quality-dataset). The features include pollutant concentrations (e.g., CO, NMHC, NOx, NO₂) and sensor measurements.

## Model Design

### Preprocessing Pipeline
A preprocessing pipeline was created to handle missing values and standardize the features. The pipeline includes:
1. **SimpleImputer**: Missing values were imputed using the mean value of the respective feature.
2. **StandardScaler**: Standardized the features to have a mean of 0 and a standard deviation of 1.

### Input Features
- Various gas/pollutants concetrations and their accompanying sensor measurements.

### Target Variable
- NO₂ concentration at the next timestamp.

### Model Selection
The **Random Forest Regressor** was chosen due to its ability to:
- Handle non-linear relationships between features.
- Perform well without extensive feature scaling or normalization.
- Provide feature importance scores for interpretability.

### Hyperparameters
The following hyperparameters were optimized for the Random Forest model:
- **Number of Trees (`n_estimators`)**: Set to 100 for a balance between performance and computation time.
- **Maximum Depth (`max_depth`)**: Limited to 10 to prevent overfitting.
- **Minimum Samples Split (`min_samples_split`)**: 2-21.
- **Minimum Samples Leaf (`min_samples_leaf`)**: 1-11.
- **Random State**: Fixed for reproducibility.

A **Randomized Grid Search** was conducted to tune these hyperparameters, using the **negative mean squared error (neg_mean_squared_error)** as the scoring metric.

## Results

### Performance Metrics
The model was evaluated using:
- **Mean Squared Error (MSE)**: 206.5233
- **Root Mean Squared Error (RMSE)**: 14.3709
- **R² Score**: 0.8902

| Metric | Value       |
|--------|-------------|
| MSE    | 206.5233    |
| RMSE   | 14.3709     |
| R²     | 0.8902      |


