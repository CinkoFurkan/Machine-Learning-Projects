# London Real Estate Price Prediction

This project predicts real estate prices in London using data from a Kaggle dataset containing property features and prices. The dataset includes details such as property type, size, number of bedrooms and bathrooms, and other key information. Various machine learning models were employed to predict property prices, and their performance was evaluated using multiple metrics.

## Dataset

The dataset used in this project is available on Kaggle: [Real Estate Data London 2024](https://www.kaggle.com/datasets/kanchana1990/real-estate-data-london-2024).

## Project Structure

- **`dataset_download()`**: Downloads the Kaggle dataset.
- **Data Preprocessing**:
  - Fills missing values in numerical columns.
  - Cleans the `price` column to ensure it's numeric.
  - Encodes categorical columns.
  - Standardizes numerical features.
  - Removes outliers from the price column.
- **Exploratory Data Analysis** (EDA):
  - Correlation analysis of numerical features.
  - Visualization of property types and price distribution.
- **Machine Learning Models**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

## Results

| Model              | MAE         | MSE           | RMSE         | R²      |
|--------------------|-------------|---------------|--------------|---------|
| Random Forest      | 2,078,469   | 8.25e+12      | 2,871,847    | 0.427   |
| Linear Regression  | 2,713,429   | 1.28e+13      | 3,577,871    | 0.111   |
| Gradient Boosting  | 2,179,747   | 8.48e+12      | 2,912,375    | 0.411   |


