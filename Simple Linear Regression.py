from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
diabetes = datasets.load_diabetes() 
X = diabetes.data 
y = diabetes.target 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, 
test_size=0.3, 
random_state=42 
) 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
model = LinearRegression() 
model.fit(X_train_scaled, y_train) 
y_pred = model.predict(X_test_scaled) 
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}") 
print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")
