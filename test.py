from ForwardForward import ForwardForward as FF
import numpy as np

X, y = np.load('X.npy'), np.load('y.npy')


model = FF(X.shape, y.shape, [20, 15, 5], 'L2Loss', 'SGD', 0.01, False)
model.train(100, X, y)

# X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
# pred = model.predict(X_test, y_test)
# print("Test Accuracy: {}".format(np.mean(pred == y_test)))
