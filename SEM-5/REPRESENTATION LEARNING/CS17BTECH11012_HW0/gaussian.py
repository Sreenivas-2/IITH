import numpy as np
import matplotlib.pyplot as plt


n = int(input("Enter the number of Samples : "))
X = np.random.normal(1.51, 0.07, n)

#mu_mle = mean and sigma_sq_mle is the variance
mu_mle = X.mean()
sigma_sq_mle = np.sum(np.square(X - X.mean()))/n

Y = np.random.normal(mu_mle, sigma_sq_mle, n)

fig, axes = plt.subplots(1, 2)
axes[0].hist(X, bins=50)
axes[1].hist(Y, bins=50)
plt.show()