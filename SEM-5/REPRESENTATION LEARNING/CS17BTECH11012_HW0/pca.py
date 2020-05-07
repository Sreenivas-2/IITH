import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
im = (Image.open("img-3.png"))
pixels = np.array(list(im.getdata()),dtype=np.float64).T
X = pixels

d = X.shape[0]
n = X.shape[1]
X -= X.mean(axis = 0)
CovX = np.cov(X)
print("Covariance Matrix of X : \n", CovX)
eigenvalues, eigenvectors = np.linalg.eig(CovX)
print("Eigen Values : \n", eigenvalues)
print("Eigen Vectors : \n", eigenvectors)
Y = np.dot(eigenvectors.T, X)
CovY = np.cov(Y)
print("Covariance Matrix of Y : \n", CovY)

#Here the Covariance Matrix of Y is a diagonal matrix which states the PCA has reduced the dimension

