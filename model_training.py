# loading necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

df=pd.read_csv(r'C:\Users\Abraham\Desktop\DOINGS\Housing.csv')

# Convert 'yes'/'no' categorical variables to binary (1/0)
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({"yes": 1, "no": 0}))

# One-hot encode 'furnishingstatus' (convert categorical to dummy variables)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# Define Features (X) and Target (y)
X = df.drop(columns=["price"])
y = df["price"]

# Convert all columns to numeric type (handling potential data type issues)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define independent variables (features) and dependent variable (target)
X = df.drop(columns=["price"])  # Features (all except target)
y = df["price"]  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Scale numerical features (important for regression stability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building: Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions on train and test sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Residual Analysis (Train Data)
residuals = y_train - y_train_pred

# Print residual summary
print("\nResiduals Summary (Train Data):")
print(residuals.describe())

# Plot Residuals Distribution
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Residuals Distribution (Train Data)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Model Evaluation
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Feature Importance using Linear Regression Coefficients
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# Sort by absolute importance
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(by="Abs_Coefficient", ascending=False)

# Display top important features
print("\nTop Features Influencing House Prices:")
print(feature_importance[["Feature", "Coefficient"]])

# Visualizing Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance, palette="coolwarm")
plt.title("Feature Importance (Linear Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

print("TO GOD BE THE GLORY")