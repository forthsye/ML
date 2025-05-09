# ML Practicals Explained with Simple Code, Manual Datasets, and Line-by-Line Explanation

## 1. Naïve Bayes Classifier (Explained)

import pandas as pd  # Importing pandas for handling data in tabular form
from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.naive_bayes import GaussianNB  # Importing Gaussian Naive Bayes model
from sklearn.metrics import accuracy_score  # For checking accuracy of predictions

# Manually created dataset with features and label
# Outlook: Sunny=0, Overcast=1, Rainy=2
# Windy: False=0, True=1
# Play: No=0, Yes=1

data = {
    'Outlook': [0, 0, 1, 2, 2, 2, 1, 0],
    'Temperature': [85, 80, 83, 70, 68, 65, 72, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 90, 80],
    'Windy': [0, 1, 0, 0, 0, 1, 1, 0],
    'Play': [0, 0, 1, 1, 1, 0, 1, 1]
}

# Converting dictionary to a DataFrame
df = pd.DataFrame(data)
X = df.drop('Play', axis=1)  # Features
y = df['Play']  # Target variable

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the model
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating accuracy
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 2. Simple Linear Regression (Explained)

```python
import pandas as pd  # For data handling
from sklearn.linear_model import LinearRegression  # Importing Linear Regression model
from sklearn.model_selection import train_test_split  # To split the data
from sklearn.metrics import mean_squared_error, r2_score  # For evaluation

# Creating a dataset: Hours studied vs Marks scored
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

# Converting to DataFrame
df = pd.DataFrame(data)
X = df[['Hours']]  # Independent variable
y = df['Marks']  # Dependent variable

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating
print("Predicted Marks:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 3. Multiple Linear Regression (Explained)

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset with two independent variables: Hours and Practice
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Practice': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Marks': [35, 40, 48, 55, 62, 68, 74, 80, 86, 92]
}

df = pd.DataFrame(data)
X = df[['Hours', 'Practice']]  # Two features
y = df['Marks']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Marks:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 4. Polynomial Regression (Explained)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Input features and target
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([35, 38, 44, 52, 60, 68, 77, 85, 90, 94])

# Converting to polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Marks:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 5. Logistic Regression (Explained)

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Binary classification example: Hours studied vs Pass/Fail

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass':           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Target label
}

df = pd.DataFrame(data)
X = df[['Hours_Studied']]
y = df['Pass']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## 6. Decision Tree Classifier (Explained)

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset

data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': [30, 40, 50, 60, 70, 80, 90, 100],
    'Student': [0, 0, 1, 1, 1, 0, 0, 1],
    'Credit_rating': [1, 1, 0, 0, 1, 1, 0, 0],
    'Buys_computer': [0, 0, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
X = df.drop('Buys_computer', axis=1)
y = df['Buys_computer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 7. Artificial Neural Network (Explained)

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR pattern

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
