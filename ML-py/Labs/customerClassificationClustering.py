def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
import pandas as pd

np.random.seed(0)

#used for making random clusters
X, y = make_blobs(n_samples = 5000, cluster_std= 0.9 ,centers= [[4,4] , [-2,1], 
                                                                [2,-3], [1,1]])

plt.scatter(X[:, 0], X[:, 1], marker = 'x')

k_means = KMeans(init = "k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(6,4))

colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))


ax = fig.add_subplot(1,1,1)

for k, col in zip(range(4), colors):
    my_members = (k_means_labels == k)
    
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())

plt.show()


k_means3 = KMeans(init = "k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)

k_means3_labels = k_means3.labels_
k_means3_center_points = k_means3.cluster_centers_
fig = plt.figure(figsize=(6,4))

colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means3_labels))))

ax = fig.add_subplot(1,1,1)
for k,col in zip(range(3), colors):
    my_members = (k_means3_labels == k)
    
    cluster_center = k_means3_center_points[k]
    
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
   
ax.set_title("KMeans3 plot")
ax.set_xticks(())
ax.set_yticks(())

plt.show()



data = pd.read_csv("Cust_Segmentation.csv")
data.head()

cust_data = data.drop("Address",axis=1)
cust_data.head()

from sklearn import preprocessing
X = cust_data.values[:,1:]
X = np.nan_to_num(X)

Clus_dataset = preprocessing.StandardScaler().fit_transform(X)

clusterNums = 3

kmeans = KMeans(n_clusters=clusterNums, n_init=12, init = 'k-means++')
kmeans.fit(X)
kmeans_labels = kmeans.labels_

kmeans_labels


cust_data["Clus_km"] = kmeans_labels
cust_data.head(5)

cust_data.groupby('Clus_km').mean()

from sklearn import preprocessing
X = cust_data.values[:,1:]
X = np.nan_to_num(X)

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=kmeans_labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()
X, y = make_blobs(n_samples=5000, n_features=4, centers=[[4, 4, 4, 4], [-2, 1, -2, 1], [2, -3, 2, -3], [1, 1, 1, 1]], cluster_std=0.9)
clusterNums = 3

kmeans = KMeans(n_clusters=clusterNums, n_init=12, init = 'k-means++')
kmeans.fit(X)
kmeans_labels = kmeans.labels_

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
plt.cla()

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.set_title("Kmeans 3d plot")
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= kmeans_labels.astype(float))

plt.show()