import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')

# Convert columns to numeric, forcing errors to NaN
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Check for and remove NaN values
df = df.dropna(subset=['price', 'engine_capacity', 'horse_power', 'cylinder', 'top_speed'])

# Split the data into training (60%), validation (20%), and test (20%) sets
train, temp = train_test_split(df, test_size=0.4, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Select features and target for all splits
X_train = train[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']].values
y_train = train['price'].values

X_val = validation[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']].values
y_val = validation['price'].values

X_test = test[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']].values
y_test = test['price'].values

# Closed-form solution implementation
def closed_form_solution(X, y):
    # Add bias (intercept) term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Compute closed-form solution: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta

# Train the model using the closed-form solution
theta = closed_form_solution(X_train, y_train)

# Predict function
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return X_b @ theta

# Predictions on validation and test sets
y_val_pred = predict(X_val, theta)
y_test_pred = predict(X_test, theta)

# Evaluate the model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Validation evaluation
val_mse, val_r2 = evaluate_model(y_val, y_val_pred)
print(f"Validation MSE: {val_mse:.2f}, R2: {val_r2:.2f}")

# Test evaluation
test_mse, test_r2 = evaluate_model(y_test, y_test_pred)
print(f"Test MSE: {test_mse:.2f}, R2: {test_r2:.2f}")

# Display the model parameters
print("Model parameters (theta):")
print(theta)

import matplotlib.pyplot as plt

# Best fit line plot for a single feature (e.g., engine_capacity)
def plot_best_fit_line(X, y, feature_index, feature_name, theta):
    # Extract the feature of interest
    feature = X[:, feature_index]
    
    # Generate predictions using the full model
    y_pred = predict(X, theta)
    
    # Sort the data for smooth plotting
    sorted_indices = np.argsort(feature)
    feature_sorted = feature[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Plot the data points
    plt.scatter(feature, y, color='blue', label='Actual Data')
    
    # Plot the best fit line
    plt.plot(feature_sorted, y_pred_sorted, color='red', label='Best Fit Line')
    plt.title(f"Best Fit Line: {feature_name} vs Price")
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.legend()
    plt.show()



# Plot Actual vs Predicted for Validation Set
def plot_actual_vs_predicted(y_actual, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r-', lw=2, label='Ideal Fit')
    plt.title(title)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate the plot for the validation set
plot_actual_vs_predicted(y_val, y_val_pred, title="Actual vs Predicted (Validation Set)")
