
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('insurance.csv')

print("Dataset Shape:", df.shape)

# EDA
numeric_columns = ['age', 'bmi', 'children', 'charges']

# Data Cleaning
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)
df_cleaned.dropna(inplace=True)

# Encoding categorical variables
df_cleaned['sex'] = df_cleaned['sex'].map({
    'male': 0,
    'female': 1
})

df_cleaned['smoker'] = df_cleaned['smoker'].map({
    'no': 0,
    'yes': 1
})

df_cleaned.rename(
    columns={
        'sex': 'is_female',
        'smoker': 'is_smoker'
    },
    inplace=True
)

# Feature Engineering
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# Select final features (region removed)
final_df = df_cleaned[
    ['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges']
].copy()

X = final_df.drop('charges', axis=1)
y = final_df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Scaling
scaler = StandardScaler()

numeric_features = ['age', 'bmi', 'children']

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numeric_features] = scaler.fit_transform(
    X_train[numeric_features]
)

X_test[numeric_features] = scaler.transform(
    X_test[numeric_features]
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nModel Evaluation")
print("----------------------")
print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Save model and scaler
joblib.dump(model, 'insurance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and scaler saved successfully!")
