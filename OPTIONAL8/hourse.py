import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')

# Convert columns to numeric, forcing errors to NaN
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Check for NaN values in the dataset
print("Number of NaN values in each column:")
print(df.isna().sum())

# Remove rows with NaN values in key columns
df = df.dropna(subset=['price', 'engine_capacity', 'horse_power', 'cylinder', 'top_speed'])


# Split the data into training (60%), validation (20%), and test (20%) sets
train, temp = train_test_split(df, test_size=0.4, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Select features and target for all splits
X_train = train[['engine_capacity', 'horse_power', 'top_speed', 'price']]
y_train = train['horse_power']

X_val = validation[['engine_capacity', 'horse_power', 'top_speed', 'price']]
y_val = validation['horse_power']

X_test = test[['engine_capacity', 'horse_power', 'top_speed', 'price']]
y_test = test['horse_power']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Add intercept term (column of 1s)
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]  # Add intercept column
X_val_scaled = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Convert to NumPy arrays
X_train_np = X_train_scaled
y_train_np = y_train.values
X_val_np = X_val_scaled
y_val_np = y_val.values
X_test_np = X_test_scaled
y_test_np = y_test.values

# Define cost function
def cost_function(X, y, theta):
    m = len(y)
    J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
    return J

# Define gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for iteration in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (X.T.dot(errors)) / m
        theta -= alpha * gradient
        cost_history.append(cost_function(X, y, theta))
    
    return theta, cost_history

# Evaluate the model and return theta
def evaluate_model(alpha):
    # Initialize parameters for gradient descent
    theta = np.zeros(X_train_np.shape[1])  # Initialize theta with zeros, shape matches the number of features including intercept
    
    # Perform gradient descent
    theta, cost_history = gradient_descent(X_train_np, y_train_np, theta, alpha, 1500)

    # Evaluate model on validation set
    y_val_pred = X_val_np.dot(theta)

    # Check if predictions contain NaN
    if np.any(np.isnan(y_val_pred)):
        return float('inf'), float('inf'), float('inf'), cost_history, theta

    mse = mean_squared_error(y_val_np, y_val_pred)
    mae = mean_absolute_error(y_val_np, y_val_pred)
    r2 = r2_score(y_val_np, y_val_pred)
    
    return mse, mae, r2, cost_history, theta

# Define a range of learning rates to test
alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# Initialize list to store results
results = []

# Test each alpha value and evaluate the performance
for alpha in alpha_values:
    mse, mae, r2, cost_history, theta = evaluate_model(alpha)
    results.append((alpha, mse, mae, r2, theta))  # Store theta with results

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame(results, columns=['Alpha', 'MSE', 'MAE', 'R2', 'Theta'])
print(results_df)

# Find the best alpha (based on lowest MSE)
best_alpha_row = results_df.loc[results_df['MSE'].idxmin()]
best_alpha = best_alpha_row['Alpha']
best_theta = best_alpha_row['Theta']
print(f"Best alpha: {best_alpha}")
print(f"Best theta: {best_theta}")

# Generate predictions for the training, validation, and test sets using the best theta
y_train_pred = X_train_np.dot(best_theta)
y_val_pred = X_val_np.dot(best_theta)
y_test_pred = X_test_np.dot(best_theta)



# Optional: Print for training and validation sets as well
mse_train = mean_squared_error(y_train_np, y_train_pred)
mae_train = mean_absolute_error(y_train_np, y_train_pred)
r2_train = r2_score(y_train_np, y_train_pred)

mse_val = mean_squared_error(y_val_np, y_val_pred)
mae_val = mean_absolute_error(y_val_np, y_val_pred)
r2_val = r2_score(y_val_np, y_val_pred)

print(f"Training Set MSE: {mse_train}")
print(f"Training Set MAE: {mae_train}")
print(f"Training Set R²: {r2_train}\n")

print(f"Validation Set MSE: {mse_val}")
print(f"Validation Set MAE: {mae_val}")
print(f"Validation Set R²: {r2_val}")

# Plot Actual vs Predicted (Validation Set)
plt.figure(figsize=(10, 6))
plt.scatter(y_val_np, y_val_pred, alpha=0.7, color='blue', label='Validation Predictions')
plt.plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], color='red', linestyle='-', label='Ideal Fit (y=x)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f"Actual vs. Predicted Prices (Validation Set, Alpha={best_alpha})")
plt.legend()
plt.grid(True)
plt.show()

# Plot cost history for the best alpha
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history)), cost_history, color='blue', label='Cost History')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History During Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
