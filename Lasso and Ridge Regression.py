from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (Diabetes dataset)
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso Regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Train Ridge Regression model
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)

# Make predictions for both models
lasso_pred = lasso.predict(X_test_scaled)
ridge_pred = ridge.predict(X_test_scaled)

# Evaluate models
print(f"Lasso Mean Squared Error: {mean_squared_error(y_test, lasso_pred):.4f}")
print(f"Lasso R^2 Score: {r2_score(y_test, lasso_pred):.4f}")

print(f"Ridge Mean Squared Error: {mean_squared_error(y_test, ridge_pred):.4f}")
print(f"Ridge R^2 Score: {r2_score(y_test, ridge_pred):.4f}")
