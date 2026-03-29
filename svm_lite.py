#!/usr/bin/env python3
"""Support Vector Machine — simplified linear SVM with SGD."""
import sys, random

class LinearSVM:
    def __init__(self, lr=0.01, lambda_reg=0.01, epochs=1000):
        self.lr, self.lambda_reg, self.epochs = lr, lambda_reg, epochs
        self.w = self.b = None
    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0]*d
        self.b = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                margin = y[i] * (sum(self.w[j]*X[i][j] for j in range(d)) + self.b)
                if margin >= 1:
                    for j in range(d):
                        self.w[j] -= self.lr * self.lambda_reg * self.w[j]
                else:
                    for j in range(d):
                        self.w[j] -= self.lr * (self.lambda_reg * self.w[j] - y[i] * X[i][j])
                    self.b -= self.lr * (-y[i])
    def predict(self, x):
        return 1 if sum(self.w[j]*x[j] for j in range(len(x))) + self.b >= 0 else -1
    def decision_function(self, x):
        return sum(self.w[j]*x[j] for j in range(len(x))) + self.b

def test():
    random.seed(42)
    X = [[random.gauss(2,0.5), random.gauss(2,0.5)] for _ in range(20)] +         [[random.gauss(-2,0.5), random.gauss(-2,0.5)] for _ in range(20)]
    y = [1]*20 + [-1]*20
    svm = LinearSVM(lr=0.001, epochs=500)
    svm.fit(X, y)
    correct = sum(1 for i in range(40) if svm.predict(X[i]) == y[i])
    assert correct >= 35, f"Accuracy too low: {correct}/40"
    assert svm.predict([3, 3]) == 1
    assert svm.predict([-3, -3]) == -1
    print("  svm_lite: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Linear SVM classifier")
