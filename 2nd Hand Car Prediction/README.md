# Car Price Prediction

This project is designed to predict the selling price of used cars based on various features such as the car's make, model, year, mileage, engine capacity, and more. The model is built using machine learning algorithms, specifically the Random Forest Regressor, to provide predictions based on the input data.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Installation

To run this project, make sure you have Python 3.x installed. Then, you can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is a CSV file named `car_details.csv` containing the following columns:

- `name`: The name/model of the car.
- `year`: The manufacturing year of the car.
- `selling_price`: The selling price of the car.
- `km_driven`: The total distance driven by the car (in kilometers).
- `fuel`: The type of fuel used by the car (Petrol/Diesel).
- `seller_type`: The type of seller (Individual/Dealer).
- `transmission`: The type of transmission (Manual/Automatic).
- `owner`: The number of previous owners.
- `mileage`: The car's mileage (in km per liter).
- `engine`: The engine size (in CC).
- `max_power`: The maximum power of the car (in bhp).
- `seats`: The number of seats in the car.

## Preprocessing

The dataset underwent several preprocessing steps, including:

- Handling missing values by dropping rows with `NaN` values.
- Converting categorical variables (such as fuel type, transmission, and owner) to numeric using encoding techniques.
- Parsing and converting columns such as mileage, engine size, and maximum power into a usable numeric format.
- Scaling the features to improve model performance.

## Model

The Random Forest Regressor model was used to predict the selling price of the car based on the features. Here is the code snippet used to train the model:

```python
from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor()

# Train the model with the training data
model.fit(X_train, y_train)
```

The model was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Results

After training and evaluating the model, the following results were obtained:

- **Mean Absolute Error (MAE)**: 68,960.88
- **Mean Squared Error (MSE)**: 14,323,703,787.55
- **Root Mean Squared Error (RMSE)**: 119,681.68
- **Model R² Score**: 0.981

The model achieved a high R² score, indicating that it explains a large portion of the variance in the car's selling price.

## Usage

To use the model for predicting car prices, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies with `pip install -r requirements.txt`.
3. Run the script to load the dataset, preprocess it, train the model, and make predictions.
4. For making predictions on new data, use the following code:

```python
y_pred = model.predict(new_data)
```
