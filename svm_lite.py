#!/usr/bin/env python3
"""Simple linear SVM (sub-gradient descent)."""
import sys, random, math
random.seed(42)
def svm_train(X,y,lr=0.01,epochs=1000,C=1.0):
    d=len(X[0]); w=[0]*d; b=0
    for _ in range(epochs):
        for xi,yi in zip(X,y):
            margin=yi*(sum(w[j]*xi[j] for j in range(d))+b)
            if margin<1:
                for j in range(d): w[j]+=lr*(yi*xi[j]-2/len(X)*w[j])
                b+=lr*yi
            else:
                for j in range(d): w[j]-=lr*2/len(X)*w[j]
    return w,b
X=[[2,3],[1,1],[3,4],[6,5],[7,8],[8,6]]
y=[1,1,1,-1,-1,-1]
w,b=svm_train(X,y)
print(f"SVM: w=[{w[0]:.3f},{w[1]:.3f}], b={b:.3f}")
for xi,yi in zip(X,y):
    pred=1 if sum(w[j]*xi[j] for j in range(2))+b>0 else -1
    print(f"  {xi} true={yi:+d} pred={pred:+d} {'✓' if pred==yi else '✗'}")
