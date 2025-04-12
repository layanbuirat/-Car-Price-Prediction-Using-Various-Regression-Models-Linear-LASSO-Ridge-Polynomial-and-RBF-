import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Load your dataset into the 'df' DataFrame (replace with your actual file path)
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')

# Ensure that the relevant columns are numeric
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['engine_capacity', 'horse_power', 'top_speed', 'price', 'cylinder'], inplace=True)

# Select features and target variable
X = df[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']]
y = df['price']

# Split the data into training (60%), validation (20%), and test sets (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Feature Selection with Forward Selection
selected_features = []  # Start with an empty list for features
remaining_features = X.columns.tolist()  # All features available for selection
best_score = float('inf')  # Initialize the best score to infinity
threshold = 0.001  # You can adjust this threshold to control when to stop

while remaining_features:
    scores_with_candidates = []
    for feature in remaining_features:
        features_to_try = selected_features + [feature]
        # Train Ridge Regression with the selected features
        ridge = Ridge(alpha=0.1)  # Initial alpha value
        ridge.fit(X_train[features_to_try], y_train)
        
        # Evaluate the model on the validation set
        y_pred_val = ridge.predict(X_val[features_to_try])
        mse_val = mean_squared_error(y_val, y_pred_val)
        scores_with_candidates.append((mse_val, feature))
    
    # Sort by MSE and select the feature with the lowest MSE
    scores_with_candidates.sort()
    best_score, best_candidate = scores_with_candidates[0]
    
    # If improvement is significant, add the feature to the model
    if best_score < float('inf') - threshold:
        selected_features.append(best_candidate)
        remaining_features.remove(best_candidate)
    else:
        break  # Stop if no further improvement is found

# After feature selection, we will use GridSearchCV to find the best alpha
print(f"Selected features: {selected_features}")

# Set up parameter grid for GridSearchCV to find the best alpha
param_grid = {'alpha': np.logspace(-4, 4, 100)}  # Alpha values from 0.0001 to 10000
ridge = Ridge()

# Perform GridSearchCV with the selected features
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train[selected_features], y_train)

# Extract the best alpha from GridSearchCV
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha found: {best_alpha}")

# Train Ridge Regression using the best alpha
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train[selected_features], y_train)

# Make predictions with the best model
y_pred_val_best = ridge_best.predict(X_val[selected_features])
y_pred_test_best = ridge_best.predict(X_test[selected_features])

# Calculate MSE and R^2 score for the best model
mse_val_best = mean_squared_error(y_val, y_pred_val_best)
mse_test_best = mean_squared_error(y_test, y_pred_test_best)
r2_val_best = r2_score(y_val, y_pred_val_best)
r2_test_best = r2_score(y_test, y_pred_test_best)

# Calculate MAE for the best model
mae_val_best = mean_absolute_error(y_val, y_pred_val_best)
mae_test_best = mean_absolute_error(y_test, y_pred_test_best)

# Print final MSE, MAE, R^2 for the model with the best alpha
print(f"Best Model - Validation MSE: {mse_val_best}")
print(f"Best Model - Test MSE: {mse_test_best}")
print(f"Best Model - Validation MAE: {mae_val_best}")
print(f"Best Model - Test MAE: {mae_test_best}")
print(f"Best Model - Validation R^2: {r2_val_best}")
print(f"Best Model - Test R^2: {r2_test_best}")

# Visualization: Feature Importances (Coefficients)
plt.figure(figsize=(8, 6))
coefficients_best = ridge_best.coef_
plt.barh(selected_features, coefficients_best, color='lightblue')
plt.title('Ridge Feature Importances (Best Alpha)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(True)
plt.show()

# Visualization: Error Distribution (Residuals)
residuals_best = y_val - y_pred_val_best
plt.figure(figsize=(8, 6))
sns.histplot(residuals_best, kde=True, color='darkblue')
plt.title('Residuals Distribution (Validation Set) - Best Alpha')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualization: Predictions vs. Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_val_best, color='blue', alpha=0.6, label='Validation Set (Best Alpha)')
plt.scatter(y_test, y_pred_test_best, color='orange', alpha=0.6, label='Test Set (Best Alpha)')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Best Model Predictions vs. Actual Values')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
