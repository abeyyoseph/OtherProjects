# SVM Classifier for Cardiovascular Disease Prediction

This project uses a **Support Vector Machine (SVM)** classifier to predict the presence of cardiovascular disease based on medical patient records. The target column, `cardio`, indicates whether a patient has cardiovascular disease (`1`) or not (`0`). 

## Dataset
Utilized the cardiovascular disease dataset from Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data  

The dataset includes the following features:
- **Numerical Features**: Age, height, weight, etc.
- **Binary Features**: Smoking, drinking, physical activity.
- **Ordinal Features**: Cholesterol levels (1: normal, 2: above normal, 3: well above normal).
- Analyzed the dataset and found no missing data.  
- Removed the patient "id" field since it would provide no useful predictive information.  

### Class Distribution
- `cardio = 0`: 35,021 samples  
- `cardio = 1`: 34,979 samples  
The dataset is highly balanced, so a stratified train-test split was not strictly necessary.

## Preprocessing Steps

1. **Scaling Numerical Features**:
   - Numerical features were standardized using the **StandardScaler** to ensure the SVM kernel operates effectively.

2. **Handling Categorical Features**:
   - Binary features (e.g., smoking) were kept as-is without scaling.
   - Ordinal features (e.g., cholesterol levels) were left unchanged, as they were already in integer format.

## Model Design Decisions

1. **Classifier**:  
   - SVM was chosen for its ability to handle high-dimensional feature spaces and deliver robust performance on tabular data.

2. **Kernel and Hyperparameter Tuning**:  
   - A **Randomized Search Grid** was used to explore the hyperparameter space:
     - `C`: Regularization parameter (`log-uniform` distribution).
     - `kernel`: `['linear', 'rbf']`
     - `gamma`: Kernel coefficient for the `rbf` kernel (`log-uniform` distribution).

   - The best hyperparameters were selected based on the **F1 score** to balance precision and recall.

3. **Evaluation Metrics**:  
   - **Accuracy**: Overall correctness of predictions.
   - **F1 Score**: Harmonic mean of precision and recall, chosen due to its sensitivity to class imbalance.

## Results

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| **SVM**       | 0.7285   | 0.7211   |
| **Naive Bayes** | 0.5867   | 0.4090   |

- The SVM model outperformed the base Naive Bayes classifier significantly.



