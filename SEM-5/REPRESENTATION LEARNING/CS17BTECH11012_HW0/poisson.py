import numpy as np
import matplotlib.pyplot as plt


n = int(input("Enter the number of Samples : "))
X = np.random.poisson(5, n)

lambda_mle = X.mean()
Y = np.random.poisson(lambda_mle, n)

fig, axes = plt.subplots(1, 2)
axes[0].hist(X, bins=15)
axes[1].hist(Y, bins=15)
plt.show()