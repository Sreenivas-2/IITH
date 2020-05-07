import numpy as np
import pandas as pd
from scipy import special
from sklearn import datasets
import gzip
import matplotlib.pyplot as plt

# X = np.zeros((300,28*28))
# y = np.zeros((300,28*28))

# f = gzip.open('/home/sai/Downloads/train-images-idx3-ubyte.gz','r')
# image_size = 28
# num_images = 500
# f.read(16)
# buf = f.read(image_size * image_size * num_images)
# data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# data = data.reshape(num_images, image_size, image_size, 1)
# x_test = np.zeros((5,784))
# x_test[0,:] = data[412].flatten()/255
# x_test[1,:] = data[437].flatten()/255
# x_test[2,:] = data[387].flatten()/255
# x_test[3,:] = data[345].flatten()/255
# x_test[4,:] = data[499].flatten()/255
# Input
# for i in range(300) :
#     X[i,:] = data[i].flatten()/255
#     #image = np.asarray(data[i]).squeeze()
#     # plt.imshow(image)
#     # plt.show()

# data = datasets.load_digits().images / 16
# X = data.reshape(len(data), -1)/16
# y = X
# n = len(data)

def spilt(data, n, ratio = 0.75) :
    x_train,x_test=[],[]
    temp = n * ratio
    for i in range(n) :
        if np.random.randint(0,n) < temp :
            x_train.append(data[i])
        else:
            x_test.append(data[i])
    return np.array(x_train),np.array(x_test)

#print(X)

def sigmoid(x) :
    return 1/(1+np.exp(-x))

# // change this
def sigmoid_der(x):
    return x * (1 - x)

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.uniform(0,0.01,(n_x, n_h))
    b1 = np.random.uniform(0,0.1,(1, n_h))
    W2 = np.random.uniform(0,0.01,(n_h, n_y))
    b2 = np.random.uniform(0,0.1,(1,n_y))
    
    params = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return params

def linear_forward(A, weights, bias) :
    return sigmoid(np.dot(A, weights) + bias)

def linear_backward(Ax, Ah, Ay, param) :
    J = (Ay - y)
    dZ = np.multiply(J, sigmoid_der(Ay))
    dX = np.multiply(np.dot(dZ, param["W2"].T), sigmoid_der(Ah))
    Ah_mu = np.sum(Ah,axis = 1) / Ax.shape[0]
    KL = np.multiply(lambda_,(-(prob/Ah_mu) + ((1-prob)/(1-Ah_mu))))
    dW2 = np.dot(Ah.T, dZ)
    db2 = np.sum(dZ, axis = 0, keepdims = True)
    dW1 = np.dot(Ax.T, dX) + (1/Ax.shape[0]) * np.dot(Ax.T,sigmoid_der(Ah))
    db1 = np.sum(dX, axis = 0, keepdims = True)

    grads = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}

    return grads

def NN_Train(X, i, h, o, learning_rate) :

    epochs = 10000

    parameters = initialize_parameters(i, h, o)

    iteration = 0

    for _ in range(epochs) :

        iteration += 1

        #   Forward_Propagation
        Z = linear_forward(X, parameters["W1"], parameters["b1"])
        Y = linear_forward(Z, parameters["W2"], parameters["b2"])

        # Error
        e = np.linalg.norm(X[0]-Y[0])

        # Back_Propagation
        gradients = linear_backward(X, Z, Y, parameters)

        # Updation
        parameters["W2"] = parameters["W2"] - (gradients["dW2"] * learning_rate)
        parameters["b2"] = parameters["b2"] - (gradients["db2"] * learning_rate)
        parameters["W1"] = parameters["W1"] - (gradients["dW1"] * learning_rate)
        parameters["b1"] = parameters["b1"] - (gradients["db1"] * learning_rate)

        print("Iteration " + str(iteration) + " - Error: " + str(e))

    return parameters


def NN_Test(input_ , param) :

    Z = linear_forward(input_, param["W1"], param["b1"])
    Y = linear_forward(Z, param["W2"], param["b2"])
    return Y


# i = 64
# h = 26 
# o = 64
lambda_ = 2
prob = 0.05
# # X1 = data[0].flatten()/255
# # print(X1.shape)
# # print(X1)

# x_train, x_test = spilt(data, len(data), 0.75)
# x_train = x_train.reshape(len(x_train),64)
# x_test = x_test.reshape(len(x_test),64)
# y = x_train
# print("Training : ")

# p = NN_Train(x_train)

# print("Testing : ")


# B = np.reshape(X1,(28,28,1))
# image = np.asarray(B).squeeze()
# plt.imshow(image, cmap = "Greys")
# plt.show()


#NN_Test(x_test, p)

def func(M,a,b) :
    fig,ax = plt.subplots(2,5)
    ax = ax.flatten()
    for i in range(10):
        if i < 5 :
            
            ax[i].imshow(M[i].reshape(b,b),cmap="Greys")
        else:
            ax[i].imshow(NN_Test(M[i-5], a).reshape(b,b),cmap="Greys")

    plt.show()

i = 64
h = 26
o = 64
learning_rate = 0.001

# Loading_Data 
data = datasets.load_digits().images / 16
X, X_test = spilt(data, len(data), 0.75)
X = X.reshape(len(X),64)
y = X
X_test = X_test.reshape(len(X_test),64)
p = NN_Train(X, i, h, o, learning_rate)
func(X_test, p, 8)



