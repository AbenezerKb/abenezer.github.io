+++
date = '2025-06-01T22:05:29Z'
draft = true
title = 'K-Means Clustering'
+++

## K-Means Clustering

Clustering is an unsupervised learning technique used to group similar data points together. Clustering can be used in market segmentation, document clustering, image segmentation and image compression. K-Means is one of widely used and easy to implement clustering algorithm. The algorithm clusters data points based on then closness of the distance between the cluster centroid. It usually uses Euclidean distance to measure the distance. Though there're different ways of implementing k-means the basic steps are similar. The first step is to initialize K centroids, this is where most of the implementations differ. Next assign each data point to the nearest centroid. Recalculate centroids as the mean of assigned points
and repeat the last two steps until the centroid stops changing. I disuss the 3 mostly known implementations. 

### Forgy/Lloyd:

This is the most widely used technique. Initially k centroids are randomly selected from the points for each clusters. Then iterativelly assign each points to the cluster with the closest centroid, compute and update the centroid after assigning points. The convergence criteria is having no change in cluster assignments or when the centroids stops moving.

Import the required library to work with

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```

**Pandas**: is used for handling data manipulation.  
**Numpy**: short for numerical python is used for numerical computing  
**Matplotlib**: is a data visualization library.  
**Sklearn(Scikit-learn)**: is machine learning library framework built on top of Numpy, SciPy, and Matplotlib
<!-- I imported the kagglehub library to load the dataset we will be using from kaggle. -->


We will load the [mall customer](https://github.com/SteffiPeTaffy/machineLearningAZ/blob/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv) dataset using pandas.

```
df = pd.read_csv('Mall_Customers.csv')
```

The first 5 rows of the dataset

```
df.head()
```

![First five rows](/posts/k-means/dataset.png)

Since the dataset contains a gender column non numerical value, we will convert to numerical values by assigning 0 to Female and 1 to Male.

```
df.loc[df["Gender"] == 'Male', 'Gender'] = 1
df.loc[df["Gender"] == 'Female', 'Gender'] = 0
```

The customer id column is not useful for clustering, so we will drop it.

```
df.drop(columns=['CustomerID'], inplace=True)
```

We will use the `Annual Income (k$)`, `Gender`, `Age`, and `Spending Score (1-100)` columns for clustering. We will scale the data using StandardScaler from sklearn.

```
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
```
To see what our data looks like we will use the matplotlib library. But since our data has 4 features and we can only visually 3 dimensions, we will use PCA for dimensionality reduction.Here's the original scaled data looks like:

```
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df.to_numpy())

plt.figure(figsize=(10, 7))
for i in range(len(np.unique(final_labels))):
    plt.scatter(X_pca[final_labels == i, 0], X_pca[final_labels == i, 1], label=f'Cluster {i+1}')

plt.title('K-Means Clustering (PCA-Reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
ax = plt.axes(projection='3d')
plt.show()

plt.scatter(kmeans_sklearn.cluster_centers_[:, 0], kmeans_sklearn.cluster_centers_[:, 1],
            c='black', marker='x', s=200, linewidth=3)
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.title('Sklearn K-means Results')
plt.grid(True, alpha=0.3)
plt.figure()
```


def initialize_centroids(X, k):
    """
    Randomly select k points from X as initial centroids.
    """
    return X[np.random.choice(X.shape[0], size=k, replace=False)]

