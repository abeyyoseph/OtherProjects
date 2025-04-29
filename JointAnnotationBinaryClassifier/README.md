# Human Activity Classification Using Joint Annotations

This project aims to build a **binary classifier** to predict whether a human is performing **lawn work** or **playing a sport**, based on joint annotation data from the MPII Human Pose dataset.

## 🧑‍💻 **Project Overview**

- **Dataset**: MPII Human Pose dataset, which includes **46 columns** representing human joint annotations.
- **Classes**: 
  - **Lawn Work**
  - **Sport**
  
- **Goal**: Train a model to classify human activities as either "lawn work" or "sport" based on joint location features.
  
## 🔧 **Steps Involved**
1. **Data Preprocessing**:
   - **Stratified Train-Test Split**: Split the data into training and testing sets, ensuring both sets contain representative proportions of the two classes (`Lawn Work` and `Sport`).
   - **Feature Engineering**: Utilized the joint annotation columns, focusing on relevant features for classification.

2. **Modeling**:
   - **Random Forest Classifier** was used for classification.
   - **Pipeline** created combining preprocessing steps (including scaling and imputation) and the classifier.
   - **Hyperparameter Tuning** via **RandomizedSearchCV** with a grid of hyperparameters to optimize the model.

3. **Evaluation**:
   - Used **F1-Score (Macro Average)** as the primary evaluation metric.
   - Secondary metrics included **precision**, **recall**, and **accuracy**.

## 📝 **Results**

### Performance Metrics:
- **F1 Macro Score**: 0.719
- **Accuracy**: 77%
- **Classification Report**:
          precision    recall  f1-score   support

   Lawn Work       0.65      0.55      0.60       139
       Sport       0.82      0.87      0.84       318

    accuracy                           0.77       457
   macro avg       0.73      0.71      0.72       457
weighted avg       0.77      0.77      0.77       457


### Key Insights:
- **Sport** was classified with high precision and recall, achieving **F1=0.84**.
- **Lawn Work** detection was more challenging, with a **lower recall (55%)**, which led to a **lower F1=0.60**.
- The **macro F1 score** reflects the performance across both classes, with a **final score of 0.72**.

### Next Steps:
- Experiment with different **thresholds** for classification to improve **recall** for the "Lawn Work" class.
- Explore alternative models (e.g., **XGBoost**, **LightGBM**) to see if a boosting model can better capture the characteristics of "Lawn Work."

## Bibliography
Thanks to the creators of the MPII Human Post Dataset
```
@inproceedings{andriluka14cvpr,
    author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
    title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June}
}
```

And thanks to these authors for enhancing the readability of the dataset
```
@inproceedings{Shukla_2022_BMVC,
    author    = {Megh Shukla and Roshan Roy and Pankaj Singh and Shuaib Ahmed and Alexandre Alahi},
    title     = {VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation},
    booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
    publisher = {{BMVA} Press},
    year      = {2022},
    url       = {https://bmvc2022.mpi-inf.mpg.de/0610.pdf}
}

@inproceedings{9706805,
    author={Shukla, Megh},
    booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
    title={Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?}, 
    year={2022},
    volume={},
    number={},
    pages={2021-2030},
    doi={10.1109/WACV51458.2022.00208}
}

@inproceedings{9523037,
    author={Shukla, Megh and Ahmed, Shuaib},
    booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
    title={A Mathematical Analysis of Learning Loss for Active Learning in Regression}, 
    year={2021},
    volume={},
    number={},
    pages={3315-3323},
    doi={10.1109/CVPRW53098.2021.00370}
}
```
