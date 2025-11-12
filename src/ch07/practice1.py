from sklearn import datasets
import random
import numpy as np

def SIGMOID(x):
    return 1/(1+np.exp(-x))

# SLP function ##
def SLP_SGD(tr_X, tr_y, alpha, rep):
    # initialize w
    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(  range(1, 100), n)
    w = (np.array(w)-50)/100
    w = w.reshape(tr_X.shape[1], -1)

    # update w
    for i in range(rep):
        for k in range(tr_X.shape[0]):
            x = tr_X[k,:]
            v = np.matmul(x, w)
            y = SIGMOID(v)
            e = tr_y[k,:] - y
            w = w + alpha * np.matmul(x.reshape(4, 1), (e*y*(1-y).reshape(1, 3)))
        print("error", i, np.mean(e))
    return w

## prepare dataset ##
iris = datasets.load_iris()
X = iris.data
target = iris.target

# one hot encoding
num = np.unique(target, axis=0)
num = num.shape[0]
y_one_hot = np.eye(num)[target]

## Training (get W) ##
W = SLP_SGD(X, y_one_hot, alpha=0.5, rep=1000)

## Test ##
pred = np.zeros(X.shape[0])
total_squared_error = 0

for i in range(X.shape[0]):
    v = np.matmul(X[i,:], W)
    y_pred = SIGMOID(v)

    total_squared_error += np.sum(np.square(y_one_hot[i] - y_pred))

    pred[i] = np.argmax(y_pred)
    print("target, predict", target[i], pred[i])

final_mse_loss = total_squared_error / X.shape[0]
print("accuracy :", np.mean(pred==target))
print("Final MSE Loss :", final_mse_loss)