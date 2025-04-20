import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_regression_cost(x, y, w, b, lambda_=1):
    m, n = x.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i + 1e-15) - (1 - y[i])*np.log(1 - f_wb_i + 1e-15)
    cost = cost / m

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_ / (2 * m)) * reg_cost

    total_cost = cost + reg_cost
    return total_cost

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_db, dj_dw


def train_logistic_regression(X, y, w, b, alpha, lambda_, epochs):
    cost_history = []

    for i in range(epochs):
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, lambda_)

        # Update weights and bias
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0 or i == epochs - 1:
            cost = compute_logistic_regression_cost(X, y, w, b, lambda_)
            cost_history.append(cost)
            print(f"Epoch {i}: cost = {cost:.4f}")
    
    return w, b, cost_history


def predict(X, w, b):
    m = X.shape[0]
    y_pred = np.zeros(m)
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        y_pred[i] = 1 if f_wb_i >= 0.5 else 0
    return y_pred




X, y = make_classification(n_samples=300, n_features=4, random_state=42)
y = y.reshape(-1)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
w = np.zeros(X.shape[1])
b = 0

w, b, cost_history = train_logistic_regression(X_train, y_train, w, b, alpha=0.1, lambda_=0.1, epochs=1000)

y_pred = predict(X_test, w, b)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")

plt.plot(np.arange(0, len(cost_history)) * 100, cost_history)
plt.xlabel("transaction")
plt.ylabel("Cost")
plt.title("Cost vs. transaction")
plt.grid(True)
plt.show()
