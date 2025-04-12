import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')


# Convert columns to numeric, forcing errors to NaN
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Remove rows with NaN values in important columns
df = df.dropna(subset=['horse_power', 'price', 'engine_capacity', 'top_speed', 'cylinder'])

# Split the data into training (60%), validation (20%), and test (20%) sets
train, temp = train_test_split(df, test_size=0.4, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Select features and target for all splits
X_train_raw = train[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']]
y_train_raw = train['price']

X_val_raw = validation[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']]
y_val_raw = validation['price']

# Data Normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_val = scaler_X.transform(X_val_raw)

y_train = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
y_val = scaler_y.transform(y_val_raw.values.reshape(-1, 1)).ravel()

# RBF Kernel with Hyperparameter Tuning
rbf_model = SVR(kernel='rbf')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

grid_search = GridSearchCV(estimator=rbf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

best_rbf_model = grid_search.best_estimator_
y_pred_val_rbf = best_rbf_model.predict(X_val)

# Denormalize predictions
y_pred_val_rbf = y_pred_val_rbf.reshape(-1, 1)
y_pred_val_rbf_denorm = scaler_y.inverse_transform(y_pred_val_rbf)

# Calculate Metrics
mse_rbf = mean_squared_error(y_val_raw, y_pred_val_rbf_denorm)
mae_rbf = mean_absolute_error(y_val_raw, y_pred_val_rbf_denorm)
r2_rbf = r2_score(y_val_raw, y_pred_val_rbf_denorm)

# Results Summary
print("RBF Kernel Model Performance:")
print(f"MSE: {mse_rbf:.4f}")
print(f"MAE: {mae_rbf:.4f}")
print(f"R2: {r2_rbf:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_val_raw.values, label="Actual Prices", color="blue")
plt.plot(y_pred_val_rbf_denorm, label="RBF Kernel Prediction", color="orange")
plt.title("Validation Set: RBF Kernel Model Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
