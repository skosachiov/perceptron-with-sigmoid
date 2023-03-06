import numpy as np
import random
import time


class Perceptron:

    def __init__(self, layers, alpha, eta):
        self.alpha = alpha
        self.eta = eta
        self.L = len(layers)
        self.o = [None] * self.L
        self.W = []
        for n in range(self.L - 1):
            self.W.append(np.random.random((layers[n], layers[n+1])))

    # sigmoid
    def F(self, x): return 1.0 / (1.0 + np.exp(-self.alpha*x))
    
    # sigmoid derivative
    def dF(self, x): return 2 * self.alpha * self.F(x) * (1.0 - self.F(x))

    def forwardprop(self, x):
        self.o[0] = self.F(np.array(x).flatten())
        for n in range(self.L - 1):
            self.o[n+1] = self.F(self.o[n] @ self.W[n])
        return self.o[self.L - 1]

    def backprop(self, t):
        e = self.o[self.L - 1] - self.F(np.array(t).flatten())
        loss = 0.5 * np.sum(e**2)
        for n in reversed(range(self.L - 1)):
            d = self.dF(self.o[n+1]) * e
            dW = -self.eta * np.outer(self.o[n], d)
            self.W[n] += dW
            e = self.W[n] @ d
        return loss

    def training(self, x, t):
        self.forwardprop(x)
        return self.backprop(t)

    def save_model():
        pass

    def load_model():
        pass

if __name__ == "__main__": 

    z = [ [1,1,1], [0,0,1], [1,1,1], [1,1,1], [1,0,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1],
          [1,0,1], [0,0,1], [0,0,1], [0,0,1], [1,0,1], [1,0,0], [1,0,0], [0,0,1], [1,0,1], [1,0,1],
          [1,0,1], [0,0,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [0,0,1], [1,1,1], [1,1,1],
          [1,0,1], [0,0,1], [1,0,0], [0,0,1], [0,0,1], [0,0,1], [1,0,1], [0,0,1], [1,0,1], [0,0,1],
          [1,1,1], [0,0,1], [1,1,1], [1,1,1], [0,0,1], [1,1,1], [1,1,1], [0,0,1], [1,1,1], [1,1,1] ]
    a = []
    b = []
    for i in range(10):
        s = []
        b = [0.0]*10
        b[i] = 1.0
        for j in range(5):
            s.append(z[i + 10*j])
        a.append((s, b))

    layers = [15, 30, 30, 10]
    alpha = 2.0
    eta = 0.005
    epoch = 1000000

    p = Perceptron(layers, alpha, eta)

    items = len(a)

    for i in range(epoch * items):
        k = random.randrange(items)
        e = p.training(a[k][0], a[k][1])
        if i%1000 == 0: 
            print("LOSS =", e)
            print("DONE =", 100 * i/(epoch*items), "%" )
            print()

    np.set_printoptions(precision=3)
    for i in range(items):
        print("TEST   =", p.F(np.array(a[i][1])))
        print("RESULT =", p.forwardprop(a[i][0]))
        print()



