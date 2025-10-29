import numpy as np


# Q1
class LogisticRegression:
    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z: float) -> float:
        # 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-z))

    def _run_gradient_descent(self, X, y, n_samples):
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            error = y_predicted - y
            dW = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self._run_gradient_descent(X, y, n_samples)

    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probability(X)
        return probabilities >= threshold

    def accuracy(self, X, y):
        y_predicted = self.predict(X)
        return np.sum(y_predicted == y) / len(y)


# Q2
def normal_equation(X, y):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    try:
        XT = X.T
        XTX = XT @ X
        XTX_inv = np.linalg.pinv(XTX)
        XTy = XT @ y
        theta = XTX_inv @ XTy

        return np.round(theta.flatten(), 4).tolist()

    except Exception as e:
        print(f"An error occurred during Normal Equation calculation: {e}")
        return None


# Q3
def gradient_descent(X, y, alpha, iterations):
    m = len(y)
    y = y.reshape(-1, 1)

    n = X.shape[1]
    theta = np.zeros((n,1))
    for i in range(iterations):
        h = X @ theta
        error = h - y

        gradient = (1/m) * (X.T @ error)
        theta -= alpha * gradient

    return np.round(theta.flatten(), 4)


def run_logistic_regression_test():
    def generate_separable_data(n_samples=100):
        """Generates a perfectly linearly separable dataset (expected 100% accuracy)."""
        np.random.seed(42)
        X = np.random.randn(n_samples, 2) * 0.5  # Tighter cluster
        # Simple linear decision boundary: x1 + x2 > 0 -> class 1
        y = ((X[:, 0] + X[:, 1] + 0.5) > 0).astype(int)
        return X, y

    def generate_overlapping_data(n_samples=100):
        """Generates a non-linearly separable dataset by creating heavily overlapping clusters
           (expected less than 100% accuracy, e.g., 75%)."""
        np.random.seed(42)

        # Use a large standard deviation (scale=1.5) and small separation (means [0,0] vs [1,1])
        overlap_scale = 1.5

        # Class 0 cluster
        X0 = np.random.normal(loc=[0, 0], scale=overlap_scale, size=(n_samples // 2, 2))
        y0 = np.zeros(n_samples // 2)

        # Class 1 cluster
        X1 = np.random.normal(loc=[1, 1], scale=overlap_scale, size=(n_samples // 2, 2))
        y1 = np.ones(n_samples // 2)

        # Combine and shuffle
        X = np.vstack((X0, X1))
        y = np.hstack((y0, y1))

        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return X, y

    """Runs two full logistic regression demos: one on separable data and one on overlapping data."""
    print("--- Logistic Regression ---")

    def run_scenario(data_fn, scenario_name, learning_rate=0.1, n_iterations=3000):
        """Helper function to run a single test scenario."""
        X, y = data_fn(n_samples=100)

        # Split Data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        # Train Model
        model = LogisticRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        print(f"\n--- Scenario: {scenario_name} ---")
        print("Starting Training...")
        model.fit(X_train, y_train)

        # Evaluate
        test_accuracy = model.accuracy(X_test, y_test)

        # Print Results
        print("\n--- Model Results ---")
        print(f"Final Weights: {model.weights}")
        print(f"Final Bias: {model.bias:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return model, test_accuracy

    # Run Scenario 1: Perfectly Separable Data
    run_scenario(generate_separable_data, "PERFECTLY SEPARABLE DATA (Expected ~100%)")

    # Run Scenario 2: Overlapping Data
    run_scenario(generate_overlapping_data, "HEAVILY OVERLAPPING DATA (Expected < 100%)")


def run_normal_equation_test():

    """
    Runs predefined test cases for the normal_equation function and prints the results.
    """
    print("\n--- Normal Equation ---")

    # Test Case 1: Perfect linear fit (y = x)
    X_1 = [[1, 1], [1, 2], [1, 3]]
    y_1 = [1, 2, 3]
    expected_1 = [0.0, 1.0]

    coefficients_1 = normal_equation(X_1, y_1)

    # Check for numerical equivalence, handling the potential [-0.0] as 0.0
    is_correct_1 = np.allclose(coefficients_1, expected_1, atol=1e-4)

    print("--- Test Case 1 ---")
    print(f"Input Features (X):\n{np.array(X_1)}")
    print(f"Input Target (y): {y_1}")
    print(f"Calculated Coefficients (theta): {coefficients_1}")
    print(f"Expected Coefficients (theta): {expected_1}")
    print(f"Test Status: {'PASSED' if is_correct_1 else 'FAILED'}")

    # Test Case 2: Perfect linear fit (y = 2x)
    X_2 = [[1, 5], [1, 15], [1, 25], [1, 35]]
    y_2 = [10, 30, 50, 70]
    expected_2 = [0.0, 2.0]

    coefficients_2 = normal_equation(X_2, y_2)
    is_correct_2 = np.allclose(coefficients_2, expected_2, atol=1e-4)

    print("\n--- Test Case 2 ---")
    print(f"Input Features (X):\n{np.array(X_2)}")
    print(f"Input Target (y): {y_2}")
    print(f"Calculated Coefficients (theta): {coefficients_2}")
    print(f"Expected Coefficients (theta): {expected_2}")
    print(f"Test Status: {'PASSED' if is_correct_2 else 'FAILED'}")

    # Final summary
    if is_correct_1 and is_correct_2:
        print("\nAll tests passed successfully.")
    else:
        print("\nOne or more tests failed.")


def run_gradient_descent_test():
    print("\n--- Gradient Descent ---")

    # --- Example 1: Basic Convergence (1000 iterations) ---
    X_input = np.array([[1, 1], [1, 2], [1, 3]])
    y_input = np.array([1, 2, 3])
    alpha_input = 0.01
    iterations_input = 1000

    coefficients = gradient_descent(X_input, y_input, alpha_input, iterations_input)

    print(f"--- Example 1: Basic Convergence (1000 iter) ---")
    print(f"Input X:\n{X_input}")
    print(f"Input y: {y_input}")
    print(f"Learning Rate (alpha): {alpha_input}")
    print(f"Iterations: {iterations_input}")
    print(f"Output Coefficients: {coefficients}")
    print(f"Expected: [0.1107 0.9513]\n")

    # --- Example 2: Test Case (should converge to near [0.0, 1.0]) ---
    X_test = np.array([[1, 4], [1, 5], [1, 6]])
    y_test = np.array([4, 5, 6])
    coefficients_test = gradient_descent(X_test, y_test, 0.01, 1000)

    print(f"--- Example 2: Simple Identity Line (1000 iter) ---")
    print(f"Input X:\n[[1 4]\n[1 5]\n[1 6]]")
    print(f"Input y: [4 5 6]")
    print(f"Output Coefficients: {coefficients_test}")
    print(f"Expected: Close to [0.0, 1.0]\n")

    # --- Example 3: Deep Convergence (10000 iterations) ---
    # Using the same input as Example 1, but with more steps
    coefficients_converged = gradient_descent(X_input, y_input, 0.01, 10000)
    print(f"--- Example 3: Full Convergence (10000 iter) ---")
    print(f"Output Coefficients: {coefficients_converged}")
    print(f"Expected: [0. 1.]")


if __name__ == '__main__':
    run_logistic_regression_test()
    run_normal_equation_test()
    run_gradient_descent_test()
