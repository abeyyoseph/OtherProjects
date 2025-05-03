# Random Forest Regression Model for NO₂ Concentration Prediction

## Project Overview

This project aims to predict Nitrogen Dioxide (NO₂) concentrations using a Random Forest Regression model. The dataset contains time-series air quality data collected over one year, with 24 data points per day (from https://www.kaggle.com/datasets/dakshbhalala/uci-air-quality-dataset). The features include pollutant concentrations (e.g., CO, NMHC, NOx, NO₂) and sensor measurements.

## Goals

The primary goal of this project is to predict **Nitrogen_Dioxide_Concentration** (NO2 concentration) from a variety of environmental and sensor-based features using a **Random Forest Regression** model. Specifically, the model aims to achieve the following objectives:
1. **Preprocessing**: Clean the data, handle missing/invalid values, and prepare the features for regression.
2. **Modeling**: Use a Random Forest Regression model to predict NO2 concentration.
3. **Evaluation**: Evaluate the model's performance using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
4. **Feature Importance**: Investigate which features contribute the most to the model's predictions.

## Data Split
Since this is a regression problem and stratified splitting is typically used for classification, a binning approach was implemented to simulate stratification. The continuous target variable, Nitrogen_Dioxide_Concentration, was divided into 10 quantile-based bins using pandas.qcut.
- Applied a stratified split of the dataset into train and test using these bins, which allowed for maintaining a balanced distribution of NO₂ concentration levels across the training and testing datasets, ensuring better generalization and evaluation consistency.

### Model Selection
The **Random Forest Regressor** was chosen due to its ability to:
- Handle non-linear relationships between features.
- Perform well without extensive feature scaling or normalization.
- Provide feature importance scores for interpretability.

### Hyperparameters
The following hyperparameters were optimized for the Random Forest model:
- **Number of Trees (`n_estimators`)**: Varied from 50-300 for a balance between performance and computation time.
- **Maximum Depth (`max_depth`)**: Varied from 5-30 to prevent overfitting.
- **Minimum Samples Split (`min_samples_split`)**: 2-21.
- **Minimum Samples Leaf (`min_samples_leaf`)**: 1-11.
- **Random State**: Fixed for reproducibility.

A **Randomized Grid Search** was conducted to tune these hyperparameters, using the **negative mean squared error** as the scoring metric.

## Results

- **Mean Squared Error (MSE)**: 220.20
- **Root Mean Squared Error (RMSE)**: 14.84
- **R² Score**: 0.8871
- **Relative RMSE**: 0.0439 (~4.39%)

These results indicate that the model has performed well, with the RMSE being around 4.39% of the total range of nitrogen dioxide concentrations.

### Feature Importance
- The most important feature for predicting **Nitrogen_Dioxide_Concentration** was **Nitrogen_Oxides_Concentration**, with an importance score of **~0.7**.
- Other features had relatively low importance (all below 0.1).

## Next Steps

1. **Feature Selection**: Consider removing the less important features (those with importance < 0.1) and retraining the model to see if performance improves.
2. **Correlation Analysis**: Investigate the correlations between features to further understand the relationships between the predictors and the target variable.
3. **Hyperparameter Tuning**: Further tune hyperparameters of the Random Forest model or consider other regression models such as XGBoost or Gradient Boosting Machines.


