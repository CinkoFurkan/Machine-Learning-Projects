# **Rainfall Prediction Classifier**

## **Project Overview**

This project involves building a machine learning classifier to predict whether it will rain the following day based on a dataset from the Australian Government's Bureau of Meteorology. The goal is to apply and compare various machine learning algorithms, including Linear Regression, K-Nearest Neighbors (KNN), Decision Trees, Logistic Regression, and Support Vector Machines (SVM).

### **Key Objectives:**
1. Clean and preprocess the rainfall dataset.
2. Implement and evaluate multiple machine learning algorithms.
3. Use metrics like Accuracy Score, Jaccard Index, F1-Score, Log Loss, Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² to assess the performance of each model.
4. Compare the results of the algorithms and draw insights based on the metrics.

## **Dataset**

The dataset used in this project comes from the Australian Government's Bureau of Meteorology, which provides information about weather conditions such as temperature, humidity, wind speed, and rainfall. The dataset includes historical weather data, and the goal is to predict if it will rain tomorrow.

The dataset can be downloaded from the following source:
- [Australian Bureau of Meteorology Dataset](https://www.bom.gov.au/climate/data/)

### **Data Columns:**
- **Date**: The date of the observation.
- **MinTemp**: Minimum temperature.
- **MaxTemp**: Maximum temperature.
- **Rainfall**: The amount of rainfall (in mm).
- **WindSpeed**: The speed of the wind.
- **Humidity**: The humidity percentage.
- **Pressure**: Atmospheric pressure.
- **RainTomorrow**: Target variable, indicating whether it will rain the next day (1 = Yes, 0 = No).

## **Algorithms Used**

In this project, the following algorithms are applied:

1. **Linear Regression**: Used for regression tasks, though it is not ideal for classification problems.
2. **K-Nearest Neighbors (KNN)**: A non-parametric classification algorithm based on measuring distances between data points.
3. **Decision Trees**: A model that uses a tree-like structure to make decisions.
4. **Logistic Regression**: A classification algorithm used to predict binary outcomes, such as rain or no rain.
5. **Support Vector Machines (SVM)**: A powerful classifier that works by finding the hyperplane that best separates classes in a high-dimensional space.

## **Evaluation Metrics**

The following metrics are used to evaluate the models:

- **Accuracy Score**: Measures the percentage of correctly predicted instances.
- **Jaccard Index**: A metric used for comparing the similarity of sample sets.
- **F1-Score**: A measure of a model's accuracy, balancing precision and recall.
- **Log Loss**: Measures the uncertainty of predictions.
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.
- **R²-Score**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables.

## **Results**

Below are the results from each classifier based on the evaluation metrics:

| Model              | Accuracy Score | Jaccard Index | F1 Score | Log Loss |
|--------------------|----------------|----------------|----------|----------|
| Linear Regression  | 0.256319       | 0.115721       | 0.427130 | NaN      |
| KNN                | 0.818321       | 0.425121       | 0.596610 | NaN      |
| Decision Trees     | 0.761832       | 0.413534       | 0.585106 | NaN      |
| Logistic Regression| 0.838168       | 0.511521       | 0.676829 | 0.381038 |
| SVM                | 0.722137       | 0.361069       | 0.419326 | NaN      |

### **Analysis**
- **Logistic Regression** outperforms the other models in terms of Accuracy Score, F1 Score, and Jaccard Index, making it the best model for this particular task.
- **KNN** and **Decision Trees** also perform reasonably well, although they fall behind Logistic Regression.
- **Linear Regression** shows lower accuracy and other metrics, as it is not a suitable choice for classification tasks.
- **SVM** has a decent performance but is not the top performer for this dataset.

### **Running the Code**
1. Clone or download this repository.
2. Open the notebook `rainfall_prediction_classifier.ipynb` in Jupyter Notebook or any Python IDE that supports notebooks.
3. Ensure that the required libraries are installed and the dataset is loaded into the notebook.
4. Run the code cells sequentially to:
   - Load and clean the dataset.
   - Split the data into training and testing sets.
   - Apply each machine learning model.
   - Evaluate each model using the metrics mentioned above.
   - Display the results.

### **Dataset Preparation**
Before applying machine learning algorithms, the dataset needs to be preprocessed:
- Handling missing values (e.g., filling or removing missing data).
- Converting categorical variables into numerical formats (e.g., One-Hot Encoding for "RainTomorrow").
- Normalizing or standardizing numerical features if necessary.

## **Conclusion**

This project demonstrates the process of building, evaluating, and comparing multiple machine learning models to predict rainfall. It highlights the importance of model evaluation using multiple metrics and the selection of the most suitable model for a given task. Logistic Regression proved to be the most effective classifier in this case, but other models like KNN and Decision Trees also performed well.
