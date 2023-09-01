import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

data = pd.DataFrame({
    "Crop": ["ARHAR", "ARHAR", "ARHAR", "ARHAR", "ARHAR", "COTTON", "COTTON", "COTTON", "COTTON", "COTTON"],
    "State": ["Uttar Pradesh", "Karnataka", "Gujarat", "Andhra Pradesh", "Maharashtra", "Maharashtra", "Punjab", "Andhra Pradesh", "Gujarat", "Haryana"],
    "Cost of Cultivation (`/Hectare) A2+FL": [9794.05, 10593.15, 13468.82, 17051.66, 17130.55, 23711.44, 29047.1, 29140.77, 29616.09, 29918.97],
    "Cost of Cultivation (`/Hectare) C2": [23076.74, 16528.68, 19551.9, 24171.65, 25270.26, 33116.82, 50828.83, 44756.72, 42070.44, 44018.18],
    "Cost of Production (`/Quintal) C2": [1941.55, 2172.46, 1898.3, 3670.54, 2775.8, 2539.47, 2003.76, 2509.99, 2179.26, 2127.35],
    "Yield (Quintal/ Hectare)": [9.83, 7.47, 9.59, 6.42, 8.72, 12.69, 24.39, 17.83, 19.05, 19.9]
})

X_linear = data[["Cost of Cultivation (`/Hectare) A2+FL", "Cost of Cultivation (`/Hectare) C2", "Cost of Production (`/Quintal) C2"]].values
y_linear = data["Yield (Quintal/ Hectare)"].values

linear_model = LinearRegression()
linear_model.fit(X_linear, y_linear)

X_vis = np.random.rand(10, 3) * np.max(X_linear, axis=0)
y_vis = linear_model.predict(X_vis)

plt.scatter(X_linear[:, 0], y_linear, label="Data points", color="blue")
plt.plot(X_vis[:, 0], y_vis, label="Linear Regression Line", color="red")
plt.xlabel("Cost of Cultivation (`/Hectare) A2+FL")
plt.ylabel("Yield (Quintal/ Hectare)")
plt.title("Linear Regression Visualization")
plt.legend()
plt.show()

X_logistic, _ = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y_logistic = np.random.randint(0, 2, size=100)

logistic_model = LogisticRegression()
logistic_model.fit(X_logistic, y_logistic)

x_min, x_max = X_logistic[:, 0].min() - 1, X_logistic[:, 0].max() + 1
y_min, y_max = X_logistic[:, 1].min() - 1, X_logistic[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X_logistic[:, 0], X_logistic[:, 1], c=y_logistic, cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Visualization")
plt.show()
