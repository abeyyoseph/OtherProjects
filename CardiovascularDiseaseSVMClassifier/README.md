Utilized the cardiovascular disease dataset from Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data

-Goal was to create an SVM classification model using the dataset.  
-Analyzed the dataset and found no missing data.  
-Removed the patient "id" field since it would provide no useful predictive information.  
-The dataset was balanced well, but still used a stratified train test split on the initial data split to ensure balanced classes in the train/test data.  
-Created a hyper-parameter grid to search through that had differing values for the C, gamma, and kernels.  
-Additionally, used the StandardScaler to scale the data.  
-Used a randomized search grid with cross validation and the F1 score metric to identify the best performing hyper-parameter values.  
-Best performing model had 73% accuracy and an F1 score of 0.72.  
-Trained a simple Naive Bayes classifier to use as comparison and this performed significantly worse with a 59% accuracy and F1 score of 0.41.  
