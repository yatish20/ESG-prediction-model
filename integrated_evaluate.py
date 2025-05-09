# integrated_evaluate_poly_improved.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

df_test = pd.read_csv('test_data_integrated.csv')  # This CSV was generated by preprocess.py
feature_cols = [
    'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption', 'Waste_Management', 'Biodiversity_Impact',
    'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment', 'Safety_Incidents', 'Labor_Rights',
    'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency', 'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
]
X_test = df_test[feature_cols].values
y_test = df_test['ESG_Score'].values

def polynomial_features(X):
    n_samples, n_features = X.shape
    X_poly = X.copy()
    X_sq = X**2
    X_poly = np.concatenate((X_poly, X_sq), axis=1)
    interactions = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    if interactions:
        X_inter = np.concatenate(interactions, axis=1)
        X_poly = np.concatenate((X_poly, X_inter), axis=1)
    return X_poly

X_test_poly = polynomial_features(X_test)
print("Test expanded feature shape:", X_test_poly.shape)

# Load the training scaler parameters for polynomial features
model_params = np.load('integrated_trained_model_poly_improved.npz', allow_pickle=True)
mu_poly = model_params['mu_poly']
sigma_poly = model_params['sigma_poly']
sigma_poly[sigma_poly == 0] = 1

X_test_poly_norm = (X_test_poly - mu_poly) / sigma_poly

W = model_params['W']
b = model_params['b']

y_pred = np.dot(X_test_poly_norm, W) + b

mse = np.mean((y_pred - y_test)**2)
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print("Mean Squared Error on test set:", mse)
print("R² Score on test set:", r2)

# ---- Custom Regression Accuracy ----
def regression_accuracy(y_pred, y_true, threshold=5.0):
    correct = np.abs(y_pred - y_true) <= threshold
    return np.mean(correct)

accuracy = regression_accuracy(y_pred, y_test, threshold=6.0)
print("Custom Regression Accuracy (error <= 5):", accuracy * 100)

print("\nSample Predictions:")
for i in range(10):
    print(f"Predicted ESG: {y_pred[i]:.2f}, Actual ESG: {y_test[i]}")

# -----------------------------
# 📊 Visualization Section
# -----------------------------

# 1. Scatter Plot: Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual ESG Score')
plt.ylabel('Predicted ESG Score')
plt.title('Predicted vs Actual ESG Scores')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Residual Histogram
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='green')
plt.title('Distribution of Residuals (Actual - Predicted)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Line Plot for Sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test[:50], label='Actual ESG', marker='o')
plt.plot(y_pred[:50], label='Predicted ESG', marker='x')
plt.title('Sample ESG Predictions vs Actual (First 50)')
plt.xlabel('Sample Index')
plt.ylabel('ESG Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
