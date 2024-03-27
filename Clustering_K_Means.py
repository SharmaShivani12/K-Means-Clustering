# Implementing the K-means clustering using elbow- method

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data_set = pd.read_csv('Mall_Customers.csv')
X = data_set.iloc[:, [3, 4]].values

'''
"iloc": This is an indexing method used in pandas. iloc stands for "integer location" 
and is used to select rows and columns by their integer index positions. 
It's one of the ways to subset or slice a DataFrame.

[:, [3, 4]]: This part of the code is specifying which parts of the DataFrame you want to select:

The ':' = select all rows.
[3, 4] after the comma specifies the integer index positions of the columns to be selected. 
In Python, indexing starts at 0, so this is referring to the 4th and 3th columns of the DataFrame.

'''
# Elbow method implementataion:
from sklearn.cluster import KMeans

# Within-Cluster Sum of Squares (WCSS)
WCSS = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 43)
    kmeans.fit(X) #applies k-means clustering to the dataset X
    WCSS.append(kmeans.inertia_) #calculates and stores the within-cluster sum of squares (WCSS) for each number of clusters. The inertia_ attribute of the KMeans object gives the sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

'''
for loop:

for i in range(1, 11): iterates over a range of values from 1 to 10 (optimal number of clusters as per dataset)
Each value of i represents the number of clusters (n_clusters) to be used in that iteration of k-means clustering.

kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) creates an instance of the KMeans class.
n_clusters = i sets the number of clusters for k-means.
init = 'k-means++' uses the k-means++ algorithm for initialization, 
which can often lead to better and faster convergence.
random_state = 43 or any number of your choice ensures that the results are reproducible by setting a seed for the random number generator

'''

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 43)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

'''
Scatter Plots for Clusters:

plt.scatter(...) is called several times, each time plotting a different cluster of data points:
X[y_kmeans == 0, 0], X[y_kmeans == 0, 1] selects all points in X that have been assigned to cluster 0 (the first cluster) by k-means, and plots them on the scatter plot. X[...] is assumed to be a 2D array-like object where X[:, 0] and X[:, 1] represent two features (like 'Annual Income' and 'Spending Score').
The same pattern is repeated for clusters 1 through 4, each with different colors ('red', 'blue', 'green', 'cyan', 'magenta') and labels.
s = 100 sets the size of the markers (dots) in the scatter plot.

'''