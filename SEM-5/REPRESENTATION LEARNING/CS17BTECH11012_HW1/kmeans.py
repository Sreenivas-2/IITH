import numpy as np

Clusters = {}

def assign_cluster(A, B, d) :
    index = np.zeros(A.shape[1])
    for j in range (A.shape[1]) :
        C = np.zeros((d,B.shape[1]))
        C = B - A[:,j:j+1]
        C = np.square(C)
        index[j] = np.argmin(np.sum(C, axis = 0))
    return index

def update_clusters(A, I, z) :
    m, n = A.shape
    clus = np.zeros((m, z))
    for i in range(z) :
        Clusters.update({'cluster-'+str(i+1) : A[:,I == i]})
        clus[:,i] = np.mean(A[:,I == i], axis = 1)
    return clus

def dis(A, B, e,i) :
    C = A - B
    C = np.square(C)
    C = np.sum(C)
    if (C > e) :
        return 1
    return 0

def k_means(X, K, E = 0.00001) :
    curr_clusters = X[:,:K]
    d = X.shape[0]
    prev_clusters = np.zeros((d,K))
    Iteration = 0
    while (dis(curr_clusters,prev_clusters,E,Iteration)) :
        Clusters.clear()
        index = assign_cluster(X, curr_clusters, X.shape[0])
        prev_clusters = curr_clusters
        curr_clusters = update_clusters(X, index, K)
        Iteration += 1
    i=0
    mu = np.zeros((d,K), dtype=np.float128)
    sigma = np.zeros((K,d,d))
    for x in Clusters:
        mu[:,i] = np.sum(Clusters[x], axis = 1)/X.shape[1]
        sigma[i,:,:] = np.cov(Clusters[x])
        i+=1
    return mu, sigma
