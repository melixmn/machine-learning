import numpy as np
import matplotlib.pyplot as plt
import copy
import math

x_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40], 
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(X, y, w, b):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        f_wb -= y[i]
        total_cost += f_wb ** 2
    return total_cost / (2 * m)


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw


def gradient_descent(w_init, X, y, b_init, alpha, cost_function, gradient_function, num_iters):
    J_history = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history


initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(initial_w, x_train, y_train,
                                            initial_b, alpha, compute_cost,
                                            compute_gradient, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f}, {w_final}")

m, _ = x_train.shape
for i in range(m):
    prediction = np.dot(x_train[i], w_final) + b_final
    print(f"Prediction: {prediction:0.2f}, Target value: {y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. Iteration")
ax2.set_title("Cost vs. Iteration (Tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('Iteration Step')
ax2.set_xlabel('Iteration Step')
plt.show()
