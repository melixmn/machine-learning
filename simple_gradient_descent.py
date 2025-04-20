import numpy as np 
import matplotlib.pyplot as plt
# Compute the cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# Compute the gradients of cost w.r.t. w and b
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b     # prediction
        error = f_wb - y[i]     # error = prediction - actual
        dj_dw += error * x[i]   # gradient w.r.t. w
        dj_db += error          # gradient w.r.t. b
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Perform gradient descent
def gradient_descent(x, y, w_init, b_init, learning_rate, iterations):
    w = w_init
    b = b_init
    cost_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % 10 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

    return w, b, cost_history

# Example dataset: y = 2x
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initialization
w_init = 0
b_init = 0
learning_rate = 0.005
iterations = 300

# Run gradient descent
w_minimized, b_minimized, cost_history = gradient_descent(x, y, w_init, b_init, learning_rate, iterations)

# Final output
print(f"Final minimized values: w = {w_minimized:.4f}, b = {b_minimized:.4f}")

# cost_plot
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iteration \n learning rate=0.005 and iteration=300")
plt.show()

# Plot the data points and the best-fit line
plt.scatter(x, y, color='blue', label='Data Points')  # original (x, y)
predicted_y = w_minimized * x + b_minimized
plt.plot(x, predicted_y, color='red', label='Linear Fit')  # the fitted line

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
