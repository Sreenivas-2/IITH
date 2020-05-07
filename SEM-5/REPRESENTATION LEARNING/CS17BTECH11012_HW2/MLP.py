import numpy as np

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def derv_sigmoid(x) :
    return x * (1 - x)


# // change this
def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_parameters(n_x, n_h, n_y = 4):

    W1 = np.random.randn(n_x, n_h)
    b1 = np.zeros((1, n_h))
    W2 = np.random.randn(n_h, 1)
    b2 = np.zeros((1,1))
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return parameters

def linear_forward(A, weights, bias) :
    return sigmoid(np.dot(A, weights) + bias)

def linear_backward(Ax, Ah, Ay, param, alpha) :
    J = Ay - y
    dZ = J * sigmoid_derivative(Ay)
    dX = np.dot(dZ, param["W2"].T) * sigmoid_derivative(Ah)
    dW2 = np.dot(Ah.T, dZ) * alpha
    db2 = np.sum(dZ, axis = 0) * alpha
    dW1 = np.dot(Ax.T, dX) * alpha
    db1 = np.sum(dX, axis = 0) * alpha

    grads = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}

    return grads

Y = np.zeros((4, 1))

def NN_Train() :

    epochs = 1000

    n_x = int(input("Enter the input no of nodes : "))
    n_h = int(input("Enter the hidden no of nodes : "))

    parameters = initialize_parameters(n_x, n_h, 4)
    
    learning_rate = 0.1


    for _ in range(epochs) :

        #   Forward_Propagation

        Z = linear_forward(X, parameters["W1"], parameters["b1"])
        Y = linear_forward(Z, parameters["W2"], parameters["b2"])

        #Prediction
        prediction = Y

        for i in range(y.shape[1]):
            if Y[i][0] > 0.5 :
                Y[i][0] = 1
            else :
                Y[i][0] =  0

        #Back_Propagation
        grads = linear_backward(X, Z, prediction, parameters, learning_rate)

        parameters["W2"] -= grads["dW2"]
        parameters["b2"] -= grads["db2"]
        parameters["W1"] -= grads["dW1"]
        parameters["b1"] -= grads["db1"]

    return parameters


def NN_Test(param) :

    X = np.array([[1,1],[0,1],[1,1],[1,0]])
    Z = linear_forward(X, parameters["W1"], parameters["b1"])
    Y = linear_forward(Z, parameters["W2"], parameters["b2"])

    for i in range(y.shape[0]):
            if Y[i][0] > 0.5 :
                Y[i][0] = 1
            else :
                Y[i][0] =  0

    print(Y)


print("Model : \n XOR - 1 \n OR -  2 \n AND - 3 \n")

case = int(input("Select the Model by entering the corresponding number : "))

noise = 0
X = np.zeros((4,2))
y = np.zeros((4,1))

if case == 1 :
    noise = 0.001
    X = np.array([[0 + noise,0 + noise],[0 + noise,1 + noise],[1 + noise,1 + noise],[1 + noise,0 + noise]])
    y = np.array([[0],[1],[0],[1]])

if case == 2 :
    noise = 0.001
    X = np.array([[0 + noise,0 + noise],[0 + noise,1 + noise],[1 + noise,1 + noise],[1 + noise,0 + noise]])
    y = np.array([[0],[1],[1],[1]]) 

if case == 3 :
    noise = 0.001
    X = np.array([[0 + noise,0 + noise],[0 + noise,1 + noise],[1 + noise,1 + noise],[1 + noise,0 + noise]])
    y = np.array([[0],[0],[1],[0]]) 

epochs = 1000

parameters = NN_Train()
NN_Test(parameters)
