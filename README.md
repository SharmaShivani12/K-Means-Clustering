Here I have tried to implement K-means clustering on data set.

K-Means Clustering is a popular unsupervised machine learning algorithm used for clustering data into a predefined number of groups, known as 'k' clusters. The algorithm works by assigning data points to the nearest cluster center, with the goal of minimizing the within-cluster variance (also known as inertia). Initially, the 'k' cluster centers are chosen randomly, and the algorithm iteratively updates these centers by calculating the mean of the points within each cluster until convergence. K-Means is known for its simplicity and efficiency, especially in handling large datasets.

K-Means++ is an enhancement over the standard K-Means algorithm, addressing its initialization phase. The primary difference lies in how the initial 'k' centroids are chosen. While K-Means selects them randomly, potentially leading to suboptimal solutions, K-Means++ chooses the first centroid randomly and then selects subsequent centroids from the remaining data points with probabilities proportional to their squared distance from the nearest existing centroid. This method tends to spread out the initial centroids more evenly, leading to better convergence and potentially more optimal clustering. K-Means++ is generally preferred due to its improved initialization process, reducing the chances of getting stuck in a local minimum.

Output:


<img width="458" alt="image" src="https://github.com/SharmaShivani12/Machine_Learning/assets/116270548/0a53487f-7b17-4722-92f6-8e8d72179bdd">



This function is adapted from Machine Learning A-Z: AI, Python & R + ChatGPT Prize [2024]







