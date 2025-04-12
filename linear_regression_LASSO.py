import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load your dataset into the 'df' DataFrame 
df = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')

# Ensure that the relevant columns are numeric
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
df['horse_power'] = pd.to_numeric(df['horse_power'], errors='coerce')
df['top_speed'] = pd.to_numeric(df['top_speed'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

# Drop rows with missing values (data cleaning)
df.dropna(subset=['engine_capacity', 'horse_power', 'top_speed', 'price', 'cylinder'], inplace=True)

# Select features and target variable
X = df[['engine_capacity', 'horse_power', 'top_speed', 'cylinder']]
y = df['price']

# Split the data into training (60%), validation (20%), and test sets (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Forward selection process
remaining_features = X.columns.tolist()
selected_features = []
best_mse = float('inf')  # Start with a high error

# Start forward selection
while remaining_features:
    mse_list = []
    
    # Evaluate each feature on the validation set
    for feature in remaining_features:
        # Add the feature to the selected set
        temp_features = selected_features + [feature]
        
        # Set up GridSearchCV to find the best alpha for each feature set
        param_grid = {'alpha': np.logspace(-4, 1, 50)}  # Testing alpha values from 0.0001 to 10
        lasso = Lasso()
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train[temp_features], y_train)
        
        # Extract the best alpha value
        best_alpha = grid_search.best_params_['alpha']
        
        # Train Lasso with the best alpha
        lasso_best = Lasso(alpha=best_alpha)
        lasso_best.fit(X_train[temp_features], y_train)
        
        # Predict and calculate MSE on the validation set
        y_pred_val = lasso_best.predict(X_val[temp_features])
        mse = mean_squared_error(y_val, y_pred_val)
        
        mse_list.append((feature, mse))
    
    # Sort by MSE (smallest first)
    mse_list.sort(key=lambda x: x[1])
    
    # Choose the feature with the lowest MSE
    best_feature, best_mse_new = mse_list[0]
    
    # Add the best feature to the selected features list if it improves the MSE
    if best_mse_new < best_mse:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_mse = best_mse_new
        print(f"Adding feature: {best_feature} with MSE: {best_mse}")
    else:
        break  # Stop if adding the feature doesn't improve the MSE

# Output the selected features
print(f"Selected Features: {selected_features}")
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha found: {best_alpha}")

# Train a final model using the selected features
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train[selected_features], y_train)

# Make predictions on the test set
y_pred_test = final_lasso.predict(X_test[selected_features])

# Calculate the final MSE, MAE, and R^2 score
final_mse = mean_squared_error(y_test, y_pred_test)
final_mae = mean_absolute_error(y_test, y_pred_test)
final_r2 = final_lasso.score(X_test[selected_features], y_test)

print(f"Final MSE: {final_mse}")
print(f"Final MAE: {final_mae}")
print(f"Final R^2: {final_r2}")

# Visualization 1: Feature Importances (Coefficients)
plt.figure(figsize=(8, 6))
coefficients = final_lasso.coef_
plt.barh(selected_features, coefficients, color='skyblue')
plt.title('LASSO Feature Importances')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(True)
plt.show()

# Visualization 2: Error Distribution (Residuals)
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualization 3: Predictions vs. Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Test Set')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Model Predictions vs. Actual Values')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
