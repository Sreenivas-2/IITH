import numpy as np
import matplotlib.pyplot as plt


n = int(input("Enter the number of Samples : "))
X = np.random.exponential(0.5, n)

# Actually the parameter is lambda but the np.random.exponential accepts 1/lambda (i.e beta) and lambda_mle = 1/X.mean()
beta_mle = X.mean()
Y = np.random.exponential(beta_mle, 20000)

fig, axes = plt.subplots(1, 2)
axes[0].hist(X, bins=20)
axes[1].hist(Y, bins=20)
plt.show()