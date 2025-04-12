import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\ML_Assignment2\m.csv')
print(f"Dataset loaded. Shape: {df.shape}")

# Check unique values in 'price' column
print("Unique values in 'price' column before cleaning:")
print(df['price'].unique())

# Clean 'price' column
df['price'] = df['price'].replace(r'[^\d.]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].fillna(df['price'].mean())

# Verify the data type of 'price' column after cleaning
print(f"Column 'price' type after cleaning: {df['price'].dtype}")
print("Unique values in 'price' column after cleaning:")
print(df['price'].unique())

# Drop 'car name' column
df = df.drop(columns=['car name'])

# Define target column and features
target_column = 'price'
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove unnecessary text and ensure numerical columns
df['price'] = df['price'].replace(r'[^\d.]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].fillna(df['price'].mean())

# Filter only numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_columns]

# Define target column and features again
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split data again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
ridge = Ridge()
lasso = Lasso()
svr_rbf = SVR(kernel='rbf')
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Train regression models
ridge.fit(X_train, y_train)
ridge_y_pred = ridge.predict(X_test)

lasso.fit(X_train, y_train)
lasso_y_pred = lasso.predict(X_test)

svr_rbf.fit(X_train, y_train)
svr_rbf_y_pred = svr_rbf.predict(X_test)

poly_pipe.fit(X_train, y_train)
poly_y_pred = poly_pipe.predict(X_test)

# Evaluate models
models = {
    'Ridge': (ridge, ridge_y_pred),
    'Lasso': (lasso, lasso_y_pred),
    'SVR (RBF Kernel)': (svr_rbf, svr_rbf_y_pred),
    'Polynomial': (poly_pipe, poly_y_pred)
}

results = {}
for name, (model, y_pred) in models.items():
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'MAE': mae, 'R^2': r2}

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    print(f"MSE: {metrics['MSE']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"R^2: {metrics['R^2']}")

# Plot results
for model_name, (model, y_pred) in models.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    plt.show()

# Select the best model based on MSE
best_model_name = min(results, key=lambda x: results[x]['MSE'])
best_model, best_y_pred = models[best_model_name]

# Print the best model name
print(f"\nBest model: {best_model_name}")

# Plot feature importances if available
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {best_model_name}")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()

# Plot residuals
residuals = y_test - best_y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black')
plt.title("Error Distribution (Residuals)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Compare predictions to actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_y_pred, alpha=0.6)
plt.title(f"{best_model_name} - Model Predictions vs Actual Values")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
