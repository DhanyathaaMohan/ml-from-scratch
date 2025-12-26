import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# Create a simple dataset
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize and train model
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print("Predictions:", predictions)
print("Weights:", model.weights)
print("Bias:", model.bias)

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, predictions, color='red', label='Predicted Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Demo")
plt.legend()
plt.show()

# Plot the loss curve
plt.plot(model.losses)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.show()
