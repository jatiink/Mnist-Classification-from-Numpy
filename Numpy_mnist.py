import numpy as np


data = np.loadtxt(r"C:\Users\jatin\Desktop\Study\Data\Mnist CSV\train.csv", delimiter=',', dtype=np.int32, skiprows=1)
m, n = data.shape  # m=42000, n=785
data.shape
np.random.shuffle(data)

data_vel = data[:1000].T
X_vel = data_vel[1:]
y_vel = data_vel[0]
X_vel = X_vel / 255.

data_train = data[1000:].T
X_train = data_train[1:]
X_train = X_train/255.
y_train = data_train[0]


def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # type: ignore
    B1 = np.random.rand(10, 1) - 0.5  # type: ignore
    W2 = np.random.rand(10, 10) - 0.5  # type: ignore
    B2 = np.random.rand(10, 1) - 0.5  # type: ignore
    return W1, B1, W2, B2


def ReLu(Z):
    return np.maximum(Z, 0)


def Softmax(Z):
    return np.exp(Z)/ sum(np.exp(Z))


def ReLU_deriv(Z):
    return Z > 0


def forward(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
