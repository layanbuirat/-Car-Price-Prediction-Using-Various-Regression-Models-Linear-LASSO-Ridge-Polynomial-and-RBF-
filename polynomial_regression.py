import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Importing the necessary metrics

# Load the dataset (ensure the correct file path)
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')

# Convert columns to numeric, forcing errors to NaN
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Check for NaN values in the dataset
print(df.isna().sum())

# Remove rows with NaN values in important columns
df = df.dropna(subset=['horse_power', 'price', 'engine_capacity', 'top_speed', 'cylinder'])

# Split the data into training and test sets (80% train, 20% test)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Select features (using 'cylinder') and target ('price')
X_train_raw = train[['cylinder']]  # Features (independent variable)
y_train_raw = train['price']  # Target (dependent variable)

X_test_raw = test[['cylinder']]  # Features (independent variable) for testing
y_test_raw = test['price']  # Target (dependent variable) for testing

# Normalize the data (standardization)
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

X_train_norm = normalize(X_train_raw)
X_test_norm = normalize(X_test_raw)

# Add a column of ones to X (for the bias term)
X_train = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
X_test = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm]

# Normalize the target (y)
y_train_norm = normalize(y_train_raw)
y_test_norm = normalize(y_test_raw)

# Function to perform Polynomial Regression
def polynomial_regression(X_train, y_train, X_test, degree):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)  # Transform training data
    X_test_poly = poly.transform(X_test)        # Transform testing data

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train_poly = model.predict(X_train_poly)
    y_pred_test_poly = model.predict(X_test_poly)

    return y_pred_train_poly, y_pred_test_poly, model

# Denormalization function (returns data to original scale)
def denormalize(X, original_data):
    return X * (original_data.max() - original_data.min()) + original_data.min()

# Create a figure for subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3x3 grid for 9 degrees (2-10)
axes = axes.ravel()  # Flatten the 2D array to 1D for easy indexing

# Loop over polynomial degrees from 2 to 10
for idx, degree in enumerate(range(2, 11)):
    # Perform Polynomial Regression for the current degree
    y_pred_train_poly, y_pred_test_poly, model = polynomial_regression(X_train_raw, y_train_norm, X_test_raw, degree)

    # Denormalize the predictions
    y_pred_train_poly_denorm = denormalize(y_pred_train_poly, y_train_raw)
    y_pred_test_poly_denorm = denormalize(y_pred_test_poly, y_test_raw)

    # Plot the results for Polynomial Regression
    axes[idx].plot(X_train_raw, y_train_raw, 'bo', label="Training data")  # Plot the original training data
    axes[idx].plot(X_train_raw, y_pred_train_poly_denorm, 'r-', label=f"Poly Degree {degree} Prediction")  # Red line for predictions
    axes[idx].set_title(f"Poly Degree {degree} Prediction", fontsize=8)  # Smaller font for title
    axes[idx].set_xlabel("train", fontsize=8)  # Smaller font for x-axis label
    axes[idx].set_ylabel("predection", fontsize=8)  # Smaller font for y-axis label
    axes[idx].legend(fontsize=8)  # Smaller font for legend

    # Compute MSE, MAE, and R² for Polynomial Regression (Training)
    mse_train_poly = mean_squared_error(y_train_raw, y_pred_train_poly_denorm)
    mae_train_poly = mean_absolute_error(y_train_raw, y_pred_train_poly_denorm)
    r2_train = r2_score(y_train_raw, y_pred_train_poly_denorm)

    # Print MSE, MAE, and R² for the training set in one line
    print(f"Degree {degree}: MSE={mse_train_poly:.4f}, MAE={mae_train_poly:.4f}, R²={r2_train:.4f}")

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
