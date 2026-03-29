#!/usr/bin/env python3
"""svm_lite - Support vector machine with SMO optimizer and kernels."""
import sys, json, math, random

def linear_kernel(x, y):
    return sum(a*b for a,b in zip(x,y))

def rbf_kernel(x, y, gamma=0.5):
    return math.exp(-gamma * sum((a-b)**2 for a,b in zip(x,y)))

class SVM:
    def __init__(self, C=1.0, kernel="linear", gamma=0.5, tol=1e-3, max_iter=100):
        self.C = C; self.tol = tol; self.max_iter = max_iter
        self.kernel_fn = linear_kernel if kernel == "linear" else lambda x,y: rbf_kernel(x,y,gamma)
    
    def fit(self, X, y):
        n = len(X); self.X = X; self.y = y
        self.alpha = [0.0]*n; self.b = 0.0
        K = [[self.kernel_fn(X[i], X[j]) for j in range(n)] for i in range(n)]
        for _ in range(self.max_iter):
            changed = 0
            for i in range(n):
                Ei = sum(self.alpha[j]*y[j]*K[i][j] for j in range(n)) + self.b - y[i]
                if (y[i]*Ei < -self.tol and self.alpha[i] < self.C) or (y[i]*Ei > self.tol and self.alpha[i] > 0):
                    j = random.randint(0, n-2)
                    if j >= i: j += 1
                    Ej = sum(self.alpha[k]*y[k]*K[j][k] for k in range(n)) + self.b - y[j]
                    ai_old, aj_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]: L = max(0, self.alpha[j]-self.alpha[i]); H = min(self.C, self.C+self.alpha[j]-self.alpha[i])
                    else: L = max(0, self.alpha[i]+self.alpha[j]-self.C); H = min(self.C, self.alpha[i]+self.alpha[j])
                    if L >= H: continue
                    eta = 2*K[i][j]-K[i][i]-K[j][j]
                    if eta >= 0: continue
                    self.alpha[j] -= y[j]*(Ei-Ej)/eta
                    self.alpha[j] = min(H, max(L, self.alpha[j]))
                    if abs(self.alpha[j]-aj_old) < 1e-5: continue
                    self.alpha[i] += y[i]*y[j]*(aj_old-self.alpha[j])
                    b1 = self.b-Ei-y[i]*(self.alpha[i]-ai_old)*K[i][i]-y[j]*(self.alpha[j]-aj_old)*K[i][j]
                    b2 = self.b-Ej-y[i]*(self.alpha[i]-ai_old)*K[i][j]-y[j]*(self.alpha[j]-aj_old)*K[j][j]
                    self.b = (b1+b2)/2; changed += 1
            if changed == 0: break
    
    def predict(self, X_test):
        preds = []
        for x in X_test:
            s = sum(self.alpha[i]*self.y[i]*self.kernel_fn(self.X[i], x) for i in range(len(self.X))) + self.b
            preds.append(1 if s >= 0 else -1)
        return preds

def main():
    random.seed(42)
    X = [[1,2],[2,3],[3,3],[6,5],[7,8],[8,6]]
    y = [-1,-1,-1,1,1,1]
    print("SVM demo\n")
    for kernel in ["linear", "rbf"]:
        svm = SVM(C=1.0, kernel=kernel); svm.fit(X, y)
        preds = svm.predict(X + [[4,4],[7,7]])
        sv = sum(1 for a in svm.alpha if a > 1e-5)
        print(f"  {kernel:7s}: preds={preds}, support_vectors={sv}")

if __name__ == "__main__":
    main()
