import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor  # إضافة SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


data = pd.read_csv(r'C:\Users\hp\Desktop\ML_Assignment2\m.csv')
print(f"Dataset loaded. Shape: {data.shape}")


data.replace({'DISCONTINUED': np.nan, 'TBD': np.nan}, inplace=True)


data = data.apply(pd.to_numeric, errors='coerce')


data['price'] = data['price'].replace(r'[^\d.]', '', regex=True)
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['price'] = data['price'].fillna(data['price'].mean())  # استبدال القيم المفقودة


data['brand'] = data['brand'].astype('category').cat.codes
data['car_name'] = data['car name'].astype('category').cat.codes


data = data.drop(columns=['car name'])

X = data.drop(['price'], axis=1)  
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

##############################

poly_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge())
])


poly_pipe.fit(X_train, y_train)

print(X_train.isnull().sum())  
###############################################


models = {
    'SVR (Linear Kernel)': SVR(kernel='linear'),
    'SVR (RBF Kernel)': SVR(kernel='rbf'),
    'Gradient Descent (SGDRegressor)': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),  
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Polynomial Regression': Pipeline([('poly', PolynomialFeatures(degree=2)),
                                       ('scaler', StandardScaler()),
                                       ('ridge', Ridge())])
}


results = {}


for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    
    model.fit(X_train_scaled, y_train)
    
    
    y_pred = model.predict(X_test_scaled)
    
  
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    results[model_name] = {
        'MSE': mse,
        'MAE': mae,
        'R^2': r2
    }


best_model_name = max(results, key=lambda model: results[model]['R^2'])
best_model = models[best_model_name]


print(f"\nBest Model Based on R²: {best_model_name}")


for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    print(f"MSE: {metrics['MSE']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"R^2: {metrics['R^2']}")


for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    plt.show()


y_test_pred = best_model.predict(X_test_scaled)


mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)


print(f"\nTest Set Evaluation for Best Model:")
print(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
print(f"R-squared (R²) on Test Set: {r2_test}")
