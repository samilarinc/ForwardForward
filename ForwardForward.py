import numpy as np

class ForwardForward(object):
    def __init__(self, X_shape, y_shape, hidden_sizes:list, loss:str, opt:str, lr:float, activation:bool):
        """
        X_shape: The shape of input matrix
        y_shape: The shape of target 'column vector' or matrix (number of rows = number of class)
        hidden_sizes: List of hidden size vectors
        loss: Loss function to be used (defaulting to L2Loss)
        opt: Optimizer of weights to be used (defaulting to SGD)
        lr: Learning rate of optimizer
        activation: Whether to use activation function (defaulting to ReLU)
        """
        self.X_shape = X_shape
        self.y_shape = y_shape
        self.W = []
        self.b = []
        num_layers = len(hidden_sizes)
        prev_size = X_shape[-1]
        next_size = hidden_sizes[0]
        for i in range(num_layers):
            self.W.append(np.random.rand(prev_size, next_size))
            self.b.append(np.random.rand(next_size))
            prev_size = next_size
            if i < num_layers - 1:
                next_size = hidden_sizes[i+1]
            else:
                next_size = y_shape[-1]

        self.S1 = np.random.rand(y_shape[-1], X_shape[-1])
        self.d1 = np.random.rand(X_shape[-1])
        self.W1 = np.random.rand(X_shape[-1], X_shape[-1])
        self.b1 = np.random.rand(X_shape[-1])
        self.opt = SGD(lr)
        self.loss = L2Loss()
        self.activation = activation

    def first_layer(self, X, y):
        h1 = X @ self.W1 + self.b1
        t1 = y @ self.S1 + self.d1
        loss = self.loss.forward(h1, t1)
        self.opt.update(self.W1, loss)
        self.opt.update(self.S1, loss)
        return h1, t1

    def loop(self, X, y):
        for i in range(0, len(self.W)):
            X = X @ self.W[i] + self.b[i]
            y = y @ self.W[i] + self.b[i]
            if self.activation:
                X = self.activation.forward(X)
                y = self.activation.forward(y)
            loss = self.loss.forward(X, y)
            self.opt.update(self.W[i], loss)
        return X, y
        
    def predict(self, X, y):
        selected = X[0]
        diffs = y - selected
        return np.argmin(diffs, axis=-1)

    def forward(self, X, y):
        X, y = self.first_layer(X, y)
        X, y = self.loop(X, y)
        return self.predict(X, y)

    def train(self, epochs, X, y):
        for i in range(epochs):
            pred = self.forward(X, y)
            print("Epoch: {:2d}, Accuracy: {:3.3f}".format(i, np.mean(pred == y)))
        return pred

class SGD(object):
    def __init__(self, lr):
        self.lr = lr
    def update(self, W, dW):
        return W - self.lr * dW
        
class L2Loss(object):
    def __init__(self):
        pass
    def forward(self, y, y_hat):
        return np.sum((y - y_hat)**2)

class ReLU(object):
    def __init__(self):
        pass
    def forward(self, x):
        return np.maximum(0, x)
