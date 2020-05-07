import numpy as np
import statistics
import matplotlib.pyplot as plt


n = int(input("Enter the number of Samples : "))

X = np.random.normal(1.2, 0.76, n)

#mu_mle will be the median of the data points
lambda_mle = np.sum(abs(X - X.mean()))/n
mu_mle = statistics.median(X)

Y = np.random.normal(mu_mle, lambda_mle, n)

fig, axes = plt.subplots(1, 2)
axes[0].hist(X, bins=50)
axes[1].hist(Y, bins=50)
plt.show()