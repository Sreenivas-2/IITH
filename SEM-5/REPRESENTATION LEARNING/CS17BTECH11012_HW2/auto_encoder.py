import numpy as np
import pandas as pd
from scipy import special
from sklearn import datasets
import gzip
import matplotlib.pyplot as plt

X = np.zeros((300,28*28))
y = np.zeros((300,28*28))

f = gzip.open('train-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 500
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
x_test = np.zeros((5,784))
x_test[0,:] = data[400].flatten()/255
x_test[1,:] = data[420].flatten()/255
x_test[2,:] = data[394].flatten()/255
x_test[3,:] = data[376].flatten()/255
x_test[4,:] = data[412].flatten()/255

# Input
for i in range(300) :
    X[i,:] = data[i].flatten()/255

# data = datasets.load_digits().images
# X = data.reshape(len(data), -1)/16
# print(X.shape)
y = X
#print(X)

def sigmoid(x) :
    return 1/(1+np.exp(-x))

# // change this
def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.uniform(0,0.01,size=(n_x, n_h))
    b1 = np.random.uniform(0,0.1,size=(1, n_h))
    W2 = np.random.uniform(0,0.01,size=(n_h, n_y))
    b2 = np.random.uniform(0,0.1,size=(1,n_y))
    
    params = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return params

def linear_forward(A, weights, bias) :
    return sigmoid(np.dot(A, weights) + bias)

def linear_backward(Ax, Ah, Ay, param) :
    J = (Ay - y)
    dZ = np.multiply(J, sigmoid_derivative(Ay))
    dX = np.multiply(np.dot(dZ, param["W2"].T), sigmoid_derivative(Ah))
    dW2 = np.dot(Ah.T, dZ)
    db2 = np.sum(dZ, axis = 0, keepdims = True)
    dW1 = np.dot(Ax.T, dX)
    db1 = np.sum(dX, axis = 0, keepdims = True)

    grads = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}

    return grads

def NN_Train() :

    epochs = 10000

    parameters = initialize_parameters(i, h, o)

    learning_rate = 0.0007

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

        print("Iteration " + str(iteration)+"Error: "+ str(e))

    return parameters


def NN_Test(input_ , param) :

    Z = linear_forward(input_, param["W1"], param["b1"])
    Y = linear_forward(Z, param["W2"], param["b2"])
    return Y
    # B = np.reshape(Y*255,(28,28,1))
    # image = np.asarray(B).squeeze()
    # plt.imshow(image, cmap = "Greys")
    # plt.show()


i = 784
h = 400
o = 784

# X1 = data[0].flatten()/255
# print(X1.shape)
# print(X1)

print("Training : ")

p = NN_Train()

print("Testing : ")


# B = np.reshape(X1,(28,28,1))
# image = np.asarray(B).squeeze()
# plt.imshow(image, cmap = "Greys")
# plt.show()


NN_Test(x_test, p)


fig,ax = plt.subplots(2,5)
ax = ax.flatten()
for i in range(10):
    if i < 5 :
        
        ax[i].imshow(x_test[i].reshape(28,28),cmap="Greys")
    else:
        ax[i].imshow(NN_Test(x_test[i-5], p).reshape(28,28),cmap="Greys")

plt.show()
