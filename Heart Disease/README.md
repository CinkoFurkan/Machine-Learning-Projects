# Heart Disease Classification using Machine Learning Models

## Project Overview
This project aims to classify whether an individual has heart disease based on various health indicators using different machine learning models. The dataset used includes demographic, lifestyle, and clinical information. The models evaluated are:

1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Machine (SVM)

## Dataset Description
The dataset contains 918 entries and 12 columns:

| Column Name       | Description                                              |
|-------------------|----------------------------------------------------------|
| **Age**           | Age of the individual in years                           |
| **Sex**           | Gender of the individual (M: Male, F: Female)            |
| **ChestPainType** | Type of chest pain (ATA, NAP, ASY, TA)                   |
| **RestingBP**     | Resting blood pressure (mm Hg)                           |
| **Cholesterol**   | Serum cholesterol (mg/dL)                                |
| **FastingBS**     | Fasting blood sugar (1: >120 mg/dL, 0: otherwise)        |
| **RestingECG**    | Resting electrocardiographic results (Normal, ST, LVH)   |
| **MaxHR**         | Maximum heart rate achieved                              |
| **ExerciseAngina**| Exercise-induced angina (Y: Yes, N: No)                  |
| **Oldpeak**       | ST depression induced by exercise relative to rest       |
| **ST_Slope**      | Slope of the peak exercise ST segment (Up, Flat, Down)   |
| **HeartDisease**  | Target variable (1: Heart Disease, 0: No Heart Disease)  |

## Preprocessing Steps
1. **Categorical Variables**:
   - One-Hot Encoding was applied to the following columns: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, and `ST_Slope`.
2. **Numerical Variables**:
   - Standardized using `StandardScaler`: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`.
3. **Splitting Data**:
   - The dataset was split into training and testing sets with an 80-20 ratio.

## Machine Learning Models

### 1. Random Forest Classifier
- **Accuracy**: 88.04%
- **Classification Report**:

```
               precision    recall  f1-score   support

           0       0.85      0.87      0.86        77
           1       0.90      0.89      0.90       107

    accuracy                           0.88       184
   macro avg       0.88      0.88      0.88       184
weighted avg       0.88      0.88      0.88       184
```

- **Confusion Matrix**:
  ```
[[67 10]
 [12 95]]
```

### 2. Logistic Regression
- **Accuracy**: 85.33%
- **Classification Report**:

```
               precision    recall  f1-score   support

           0       0.80      0.87      0.83        77
           1       0.90      0.84      0.87       107

    accuracy                           0.85       184
   macro avg       0.85      0.86      0.85       184
weighted avg       0.86      0.85      0.85       184
```

- **Confusion Matrix**:
  ```
[[62 15]
 [17 90]]
```

### 3. Support Vector Machine (SVM)
- **Accuracy**: 69.02%
- **Classification Report**:

```
               precision    recall  f1-score   support

           0       0.61      0.70      0.65        77
           1       0.76      0.68      0.72       107

    accuracy                           0.69       184
   macro avg       0.69      0.69      0.69       184
weighted avg       0.70      0.69      0.69       184
```

- **Confusion Matrix**:
  ```
[[54 23]
 [34 73]]
```

## Conclusion
- **Best Model**: Random Forest Classifier achieved the highest accuracy (88.04%) and best overall performance.
- Logistic Regression also performed well, with an accuracy of 85.33%.
- Support Vector Machine had the lowest accuracy (69.02%), indicating it may not be well-suited for this dataset without further optimization.

## Future Work
- Explore hyperparameter tuning for Random Forest and Logistic Regression to further improve accuracy.
- Evaluate additional algorithms such as Gradient Boosting or XGBoost.
- Perform feature selection to identify the most important predictors of heart disease.

---
