import numpy as np
import math as mt
import matplotlib.pyplot as plt

m = int(input("Enter the number of Trials : "))
X = np.random.binomial(m, 0.73, 10000)

#p_mle will be (sum of all values in X)/(m*n) ; m : no of trials , n : no of experiments
p_mle = X.mean()/m
Y = np.random.binomial(m, p_mle, 10000)

fig, axes = plt.subplots(1, 2)
axes[0].hist(X, bins=100)
axes[1].hist(Y, bins=100)
plt.show()