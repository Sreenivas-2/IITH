import numpy as np
import random
import kmeans

def gaussian(X, mu, sigma, l, k) :
    return np.exp((-0.5)*np.dot((X[:,l].reshape((d,1))-mu[:,k].reshape((d,1))).transpose(),np.dot(np.linalg.inv(sigma[k,:,:]),(X[:,l].reshape((d,1))-mu[:,k].reshape((d,1))))))/(np.sqrt(((2*np.pi)**d)*abs(np.linalg.det(sigma[k,:,:]))))

def update_gamma(gamma, pi, mu, sigma) :
    x, y = gamma.shape
    denom=0
    for i in range(x) :
        denom=0
        for j in range(y) :
            denom += (pi[j]*gaussian(X, mu, sigma, i, j))
        for j in range(y) :
            gamma[i,j] = (pi[j]*gaussian(X, mu, sigma, i, j)) / (denom)
    return gamma

def update_mean(mu, gamma) :
    x, y = mu.shape
    Nk = np.sum(gamma, axis = 0)
    for j in range(y) :
        mu[:,j] = np.sum(gamma[:,j].reshape((n,1)).T * X, axis = 1) / Nk[j]
    return mu

def update_sigma(sigma, mu, gamma) :
    x = sigma.shape[0]
    Nk = np.sum(gamma, axis = 0)
    for i in range(x) :
        sigma[i,:,:].fill(0)
        for j in range(n) :
            sigma[i,:,:] += gamma[j,i] * ((X[:,j].reshape((d,1))-mu[:,i].reshape((d,1))).dot((X[:,j].reshape((d,1))-mu[:,i].reshape((d,1))).T))
        sigma[i,:,:] /= Nk[i]
    return sigma

def update_pi(pi, gamma) :
    Nk = np.sum(gamma, axis = 0)
    return Nk/n

def dis(A, B, e,i) :
    C = abs(A - B)
    print("Error at Iteration " + str(i) + " : ", C,"\n")
    if (C > e) :
        return 1
    return 0

def max_likelihood(X, mu, sigma, pi) :
    max=0
    for i in range(n) :
        likelihood=0
        for  j in range(K) :
            likelihood += gaussian(X, mu, sigma, i, j) * pi[j]
        max += np.log(likelihood)
    return max

def readfile(file_name):
    data = []
    fptr = open(file_name,"r")
    line = fptr.readlines()
    for l in line:
        data.append(list(map(float,l.split())))
    data = np.array(data)
    return data

def GMM(X) :
    mu , sigma = kmeans.k_means(X, K)
    pi = np.zeros((K,1))
    for i in range(K) :
        pi[i]=(1/K)
    gamma = np.zeros((X.shape[1], K))
    Iteration = 0
    curr_lik = max_likelihood(X, mu, sigma, pi)
    prev_lik = 0
    while (dis(curr_lik, prev_lik, E, Iteration)) :
        prev_lik = curr_lik
        gamma = update_gamma(gamma, pi, mu, sigma)
        mu = update_mean(mu, gamma)
        sigma = update_sigma(sigma, mu, gamma)
        pi = update_pi(pi, gamma)
        Iteration+=1
        curr_lik = max_likelihood(X, mu, sigma, pi)
    print("Mean : \n", mu,"\n")
    print("Sigma : \n", sigma,"\n")
    print("Weights : \n", pi,"\n")

filename = input("Enter the Data-FileName('input1.txt' / 'input2.txt' / custom-created-filename) : ")
X = readfile(filename)
d = X.shape[0]
n = X.shape[1]
K = int(input("Enter Mixture Size : "))
E = float(input("Enter the Threshold : "))
GMM(X)
