# Fake News Detection Project

This project aims to detect whether a news headline is real or fake using a Logistic Regression model. The dataset used is the "Fake or Real News Dataset" from Kaggle.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview
Fake news detection is a critical challenge in today's digital world. This project uses machine learning techniques to classify news headlines as either "Fake" or "Real."

## Installation
1. Clone the repository or download the script.
2. Ensure Python is installed on your machine.
3. Install required Python libraries:
   ```bash
   pip install pandas numpy nltk scikit-learn kagglehub
   ```
4. Download the dataset from Kaggle using the `kagglehub` library.

## Dataset
The dataset is downloaded from Kaggle using the `kagglehub` library:

```python
import kagglehub

path = kagglehub.dataset_download("nitishjolly/news-detection-fake-or-real-dataset")
```

The dataset contains headlines labeled as "Fake" or "Real."

## Preprocessing
Text data undergoes the following preprocessing steps:
1. **Removing special characters**
2. **Converting text to lowercase**
3. **Tokenizing text**
4. **Removing stopwords**
5. **Stemming words**

The processed text is stored in a corpus and vectorized using `CountVectorizer`.

## Model Training
The Logistic Regression model is trained on the vectorized features and their corresponding labels using an 80-20 train-test split:

```python
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train, y_train)
```

## Evaluation
The model is evaluated using accuracy and a confusion matrix:

```python
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```

## Results
- **Accuracy:** 99.75%
- **Confusion Matrix:**
  ```
  [[1013    4]
   [   1  962]]
  ```

## Acknowledgements
- [Nitish Jolly's Fake or Real News Dataset](https://www.kaggle.com/datasets/nitishjolly/news-detection-fake-or-real-dataset)
- Libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `kagglehub`

---

This project is a step towards addressing the growing challenge of fake news. Contributions and suggestions are welcome!

