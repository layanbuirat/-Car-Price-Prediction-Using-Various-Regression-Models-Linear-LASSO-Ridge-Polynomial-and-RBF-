import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor  # Gradient Descent
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# تحميل البيانات
data = pd.read_csv(r'C:\Users\Admin\Desktop\m.csv')
print(f"Dataset loaded. Shape: {data.shape}")

# استبدال القيم غير الرقمية بقيم مفقودة NaN
data.replace({'DISCONTINUED': np.nan, 'TBD': np.nan}, inplace=True)

# محاولة تحويل الأعمدة إلى أرقام
data = data.apply(pd.to_numeric, errors='coerce')

# تنظيف عمود 'price'
data['price'] = data['price'].replace(r'[^\d.]', '', regex=True)
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['price'] = data['price'].fillna(data['price'].mean())  # استبدال القيم المفقودة

# معالجة البيانات: تحويل الأعمدة النصية إلى قيم عددية
data['brand'] = data['brand'].astype('category').cat.codes
data['car_name'] = data['car name'].astype('category').cat.codes

# حذف عمود 'car name' لأنه يحتوي على نصوص ولا يستخدم في النماذج
data = data.drop(columns=['car name'])

# تحديد المدخلات (X) و المتغير التابع (y)
X = data.drop(['price'], axis=1)  # إزالة عمود السعر
y = data['price']

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# معالجة القيم المفقودة في X_train و X_test باستخدام SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# تطبيق التوسيع (scaling) على بيانات المدخلات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# إنشاء نموذج Gradient Descent باستخدام SGDRegressor
gradient_descent_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

# تدريب النموذج
gradient_descent_model.fit(X_train_scaled, y_train)

# التنبؤ باستخدام النموذج
y_pred = gradient_descent_model.predict(X_test_scaled)

# حساب مقاييس التقييم
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# طباعة نتائج التقييم
print("\nGradient Descent Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# رسم التنبؤات مقابل القيم الفعلية
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.title("Gradient Descent - Predicted vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid()
plt.show()

# توزيع الأخطاء (Residuals)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title("Error Distribution (Residuals)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()
