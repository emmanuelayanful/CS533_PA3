# **A Mathematical Study of K-Means Clustering Method**

### Author: Emmanuel Kwame Ayanful

## **Introduction**
K-Means clustering is a partitional clustering method that attempts to partition a given dataset into a number of pre-defined distinct non-overlapping subgroups where each data point belongs to one and only group.

## **The K-Means Clustering Process**
We illustrate the K-Means clustering process by considering the dataset below:

| Observations | Point_x | Point_y |
|:-----------:|:------------:|:------------:|
| A | 1 | 1 |
| B | 2 | 1 |
| C | 4 | 3 |
| D | 5 | 4 |
| E | 1 | 2 |
| F | 4 | 4 |

And as such we generate the tuples $A(1, 1), B(2, 1), C(4, 3), D(5, 4), E(1, 2), F(4, 4)$ from the above table. the process is as follows:


1.   Specify the number of clusters $k$ and choose $k$ initial centroids.
Here we want two (2) clusters so we choose the points $A$ and $B$ as our initial centroids which gives $C^1(1, 1)$ and $C^2(2, 1).$
2.   Find the sum of the squared distances between the data points and the centroids.

| Observations | X | $\|X - (1, 1)\|$ | $\| X - (2, 1)\|$ |
|:-----------:|:------------:|:------------:|:------------:|
| A | (1, 1) | $(1 - 1)^2 + (1 - 1)^2 = 0$ | $(1 - 2)^2 + (1 - 1)^2 = 1$ |
| B | (2, 1) | $(2 - 1)^2 + (1 - 1)^2 = 1$ | $(2 - 2)^2 + (1 - 1)^2 = 0$ |
| C | (4, 3) | $(2 - 1)^2 + (3 - 1)^2 = 13$ | $(4 - 2)^2 + (3 - 1)^2 = 8$ |
| D | (5, 4) | $(5 - 1)^2 + (4 - 1)^2 = 25$ | $(5 - 2)^2 + (4 - 1)^2 = 18$ |
| E | (1, 2) | $(1 - 1)^2 + (2 - 1)^2 = 1$ | $(1 - 2)^2 + (2 - 1)^2 = 2$ |
| F | (4, 4) | $(4 - 1)^2 + (4 - 1)^2 = 18$ | $(4 - 2)^2 + (4 - 1)^2 = 13$ |

3.   Assign each observation to the closest cluster on the basis of smallest distance to cluster's centroid.

| Observations | X | \|X - (1, 1)\| | \| X - (2, 1)\| | Cluster |
|:-----------:|:------------:|:--------------------------------:|:---------------------------:|:-------------:|
| A | (1, 1) | 0 | 1 | $C_1$ |
| B | (2, 1) | 1 | 0 | $C_2$ |
| C | (4, 3) | 13 | 8 | $C_2$ |
| D | (5, 4) | 25 | 18 | $C_2$ |
| E | (1, 2) | 1 | 2 | $C_1$ |
| F | (4, 4) | 18 | 13 | $C_2$ |

4.   Recompute the center of each cluster by taking the aritmetic mean of all observations belonging to a specified cluster.

$$C^{1}_{new} = \frac{A+E}{2}=\frac{(1,1)+(1,2)}{2}=\frac{(2,3)}{2}=(1,1.5)$$
$$C^{2}_{new} = \frac{B+C+D+F}{4}=\frac{(2,1)+(4,3)+(5,4)+(4,4)}{4} =\frac{(15,12)}{4}=(3.75,3)$$

5.   Repeat step 2, 3, and 4 until no data point changes cluster or centroids do not change values.

| Observations | X | $\|X - (1, 1.5)\|$ | $\| X - (3.75, 3)\|$ |
|:----:|:------:|:----------------:|:----------------:|
| A | (1, 1) | $(1-1)^2+(1-1.5)^2=0.25$ | $(1-3.75)^2+(1-3)^2=11.5625$ |
| B | (2, 1) | $(2-1)^2+(1-1.5)^2=1.25$ | $(2-3.75)^2+(1-3)^2=7.0625$ |
| C | (4, 3) | $(4-1)^2+(3-1.5)^2=11.25$ | $(4-3.75)^2+(3-3)^2=0.0625$ |
| D | (5, 4) | $(5-1)^2+(4-1.5)^2=22.25$ | $(5-3.75)^2+(4-3)^2=2.5625$ |
| E | (1, 2) | $(1-1)^2+(2-1.5)^2=0.25$ | $(1-3.75)^2+(2-3)^2=8.5625$ |
| F | (4, 4) | $(4-1)^2+(4-1.5)^2=15.25$ | $(4-3.75)^2+(4-3)^2=1.0625$ |

| Observations | X | \|X - (1, 1.5)\| | \|X - (3.75, 3)\| | Cluster |
|:----:|:------:|:----------------:|:----------------:|:---------:|
| A | (1, 1) | 0.25 | 11.5625 | $C_1$ |
| B | (2, 1) | 1.25 | 7.0625 | $C_1$ |
| C | (4, 3) | 11.25 | 0.0625 | $C_2$ |
| D | (5, 4) | 22.25 | 2.5625 | $C_2$ |
| E | (1, 2) | 0.25 | 8.5625 | $C_1$ |
| F | (4, 4) | 15.25 | 1.0625 | $C_2$ |


$$
C_{new}^1 = \frac{A+B+E}{3} = \frac{(1,1)+(2,1)+(1,2)}{3} = (1.33,\,1.33)
$$

$$
C_{new}^2 = \frac{C+D+F}{3} = \frac{(4,3)+(5,4)+(4,4)}{3} = (4.33,\,3.67)
$$

| Observations | X |  $\|X -  (1.33, 1.33)\|$ |  $\|X -  (4.33, 3.67)\|$ |
|:----:|:------:|:----------------:|:----------------:|
| A | (1, 1) | $(1-1.33)^2+(1-1.33)^2=0.218$ | $(1-4.33)^2+(1-3.67)^2=18.218$ |
| B | (2, 1) | $(2-1.33)^2+(1-1.33)^2=0.558$ | $(2-4.33)^2+(1-3.67)^2=12.558$ |
| C | (4, 3) | $(4-1.33)^2+(3-1.33)^2=9.918$ | $(4-4.33)^2+(3-3.67)^2=0.558$ |
| D | (5, 4) | $(5-1.33)^2+(4-1.33)^2=20.598$ | $(5-4.33)^2+(4-3.67)^2=0.558$ |
| E | (1, 2) | $(1-1.33)^2+(2-1.33)^2=0.558$ | $(1-4.33)^2+(2-3.67)^2=13.878$ |
| F | (4, 4) | $(4-1.33)^2+(4-1.33)^2=14.258$ | $(4-4.33)^2+(4-3.67)^2=0.218$ |



| Observations | X |  \|X -  (1.33, 1.33)\| |  \|X -  (4.33, 3.67)\| | Cluster |
|:----:|:------:|:----------------:|:----------------:|:---------:|
| A | (1, 1) | 0.218 | 18.218 | $C_1$ |
| B | (2, 1) | 0.558 | 12.558 | $C_1$ |
| C | (4, 3) | 9.918 | 0.558 | $C_2$ |
| D | (5, 4) | 20.598 | 0.558 | $C_2$ |
| E | (1, 2) | 0.558 | 13.878 | $C_1$ |
| F | (4, 4) | 14.258 | 0.218 | $C_2$ |


$$
C_{new}^1 = \frac{A+B+E}{3} = \frac{(1,1)+(2,1)+(1,2)}{3} = (1.33,\,1.33)
$$

$$
C_{new}^2 = \frac{C+D+F}{3} = \frac{(4,3)+(5,4)+(4,4)}{3} = (4.33,\,3.67)
$$

At this point, the centroids no longer change values, indicating that the K-Means algorithm has converged.

![K-Means Iteration Plot](./images/kmeans_plot.jpeg)

## **K-means As An Optimization Problem**
### **The Objective Function**
The main objective function of the K-Means algorithm is given by:

$$
\begin{aligned}
J &= \sum_{j=1}^{k}\sum_{i: x_i \in j} \|x_i - \mu_j\|^2 \\
  &= \sum_{j=1}^{K}\sum_{i=1}^{n} w_{ij}\|x_i - \mu_j\|^2
\end{aligned}
$$

where  
- $x_i$ is the $i^{th}$ data point,  
- $\mu_j$ is the center of the $j^{th}$ cluster, and  
- $w_{ij}$ is an assignment indicator defined as:

$$
w_{ij} =
\begin{cases}
1, & \text{if } x_i \text{ is assigned to cluster } j, \\
0, & \text{otherwise.}
\end{cases}
$$


The problem here is a **minimization problem** with respect to two variables.  
1. We first minimize $J$ with respect to $w_{ij}$ by treating $\mu_j$ as constant.  
2. Then, we minimize $J$ with respect to $\mu_j$ while treating $w_{ij}$ as constant.  

We aim to choose the **optimal cluster assignments** $w_{ij}$ for fixed centers $\mu_j$. This is the **Expectation Step (E-step)**.  
Next, we compute the **optimal cluster centers** $\mu_j$ for fixed assignments $w_{ij}$. This is the **Maximization Step (M-step)**.

### **The Expectation Step**

Here, we minimize $J$ by holding $\mu_k$ constant and optimizing $w_{ij}$.

$$
w_{ij} =
\begin{cases}
1, & \text{if } j = \arg\min_{l} \|x_i - \mu_l\|^2, \\
0, & \text{otherwise.}
\end{cases}
$$

That is, each data point $x_n$ is assigned to the **closest cluster** with centroid $\mu_k$, based on the **minimum squared Euclidean distance**.

### **The Maximization Step**

We now take the partial derivative of $J$ with respect to $\mu_j$:

$$
\frac{\partial J}{\partial \mu_j} =
\frac{\partial}{\partial \mu_j}
\sum_{i=1}^{n} w_{ij} \|x_i - \mu_j\|^2
$$

Expanding $\|x_i - \mu_j\|^2$:

$$
\begin{aligned}
\|x_i - \mu_j\|^2
&= (x_i - \mu_j)^T(x_i - \mu_j) \\
&= x_i^T x_i - x_i^T \mu_j - \mu_j^T x_i + \mu_j^T \mu_j \\
&= x_i^T x_i - 2x_i^T \mu_j + \mu_j^T \mu_j
\end{aligned}
$$

Substituting into the derivative of $J$:

$$
\begin{aligned}
\frac{\partial J}{\partial \mu_j}
&= \frac{\partial}{\partial \mu_j}
\sum_{i=1}^{n} w_{ij}
\left(x_i^T x_i - 2x_i^T \mu_j + \mu_j^T \mu_j\right) \\
&= \sum_{i=1}^{n} w_{ij}
\left(-2x_i + 2\mu_j\right) \\
&= -2\sum_{i=1}^{n} w_{ij}x_i + 2\mu_j \sum_{i=1}^{n} w_{ij}
\end{aligned}
$$

Setting the derivative equal to zero:

$$
\frac{\partial J}{\partial \mu_j} = 0
$$

gives

$$
-2\sum_{i=1}^{n} w_{ij}x_i + 2\mu_j \sum_{i=1}^{n} w_{ij} = 0
$$

Simplifying,

$$
\mu_j = \frac{\sum_i w_{ij} x_i}{\sum_i w_{ij}}
$$

Let

$$
\sum_i w_{ij} = n_j
$$

then

$$
\mu_j = \frac{\sum_{i: x_i \in j} x_i}{n_j}
$$

Thus, each centroid $\mu_j$ is simply the **arithmetic mean of all data points assigned to cluster $j$**.


### **Second Derivative (Convexity Check)**

For each cluster $j$, the matrix of second derivatives is given as:

$$
\begin{aligned}
\frac{\partial^2 J}{\partial \mu_j^2}
&= \frac{\partial}{\partial \mu_j}
\sum_i w_{ij}(-2x_i + 2\mu_j) \\
&= 2\sum_i w_{ij} I > 0
\end{aligned}
$$

Since the second derivative is positive, the function $J$ is **convex** with respect to $\mu_j$, confirming that the centroid update indeed minimizes the objective function.

### **Pros and Cons of the K-Means Algorithm**

#### **Pros**

Some advantages of the K-Means algorithm are outlined below:

- The K-Means algorithm is relatively simple to implement compared to other methods.  
- It produces clusters that are intuitive to interpret and visualize. Because of its simplicity, it can be very useful when you need a quick overview of data structure.  
- The K-Means approach has linear time complexity and can easily be applied to large datasets.


#### **Cons**

Some disadvantages of the K-Means algorithm are outlined below:

- The algorithm requires the user to specify the number of clusters $k$ in advance, which may be difficult without prior knowledge or domain expertise.  
- It is highly sensitive to the initialization of centroids — the presence of noise points or outliers can lead to poor clustering results.  
- It uses a distance-based similarity measure that can perform poorly in high-dimensional spaces due to the “curse of dimensionality.”  
- It does not handle clusters of varying densities and sizes well. Generalized variants of K-Means are often required for better results.


### **The Elbow Method**

In K-Means clustering, the **Elbow Method** is a parameter-tuning technique used to estimate the ideal number of clusters.  
It plots the *inertia* (or Within-Cluster Sum of Squares, WCSS) for different values of $k$.

As $k$ increases:
- inertia decreases (clusters get tighter),  
- but the improvement diminishes after a certain point.  

That point of inflection — the *“elbow”* — represents the optimal number of clusters.


### **Inertia**

Inertia measures the compactness of clusters by computing the **sum of squared deviations** between each observation and its assigned centroid.

Mathematically:

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2
$$

A good model has both:
- a **low inertia value**, and  
- a **small number of clusters ($K$)**.

However, as $K$ increases, inertia always decreases — so the Elbow Method helps balance this trade-off.


### **Step-by-Step Example**

1. Choose several candidate values for $k$, e.g., $[1, 2, 3, 4, 5]$.  
2. Compute the WCSS for each case using the formula above.  
3. Plot WCSS versus $k$ and locate the "elbow" point.


#### **Case 1: Number of Clusters = 1**

All observations belong to a single cluster with centroid $(2.8333, 2.5)$.

| Obs. | X | \|X - (2.8333, 2.5)\|^2 | Cluster |
|:----:|:------:|:----------------:|:---------:|
| A | (1, 1) | 6.1099 | $C_1$ |
| B | (2, 1) | 2.9439 | $C_1$ |
| C | (4, 3) | 1.6119 | $C_1$ |
| D | (5, 4) | 6.9459 | $C_1$ |
| E | (1, 2) | 3.6099 | $C_1$ |
| F | (4, 4) | 3.6119 | $C_1$ |
| **Total** |  | **24.3334** |  |

$\text{WCSS} = 24.3334$


#### **Case 2: Number of Clusters = 2**

Observations A, B, and E form cluster 1 with centroid $(1.3333, 1.3333)$,  
and observations C, D, and F form cluster 2 with centroid $(4.3333, 3.6667)$.

| Obs. | X | Cluster | \|X-(1.3333,1.3333)\|^2 | \|X-(4.3333,3.6667)\|^2 |
|:----:|:------:|:---------:|:----------------:|:----------------:|
| A | (1, 1) | $C_1$ | 0.222 |   |
| B | (2, 1) | $C_1$ | 0.5554 |   |
| C | (4, 3) | $C_2$ |   | 0.5556 |
| D | (5, 4) | $C_2$ |   | 0.5556 |
| E | (1, 2) | $C_1$ | 0.556 |   |
| F | (4, 4) | $C_2$ |   | 0.2222 |
| **Total** |  |  | **1.3334** | **1.3334** |

$\text{WCSS} = 1.3334 + 1.3334 = 2.6668$


#### **Case 3: Number of Clusters = 3**

- Cluster 1: A, B with centroid $(1.5, 1)$  
- Cluster 2: C, D, F with centroid $(4.3333, 3.6667)$  
- Cluster 3: E with centroid $(1, 2)$  

Cluster 1 inertia = 0.5  
Cluster 2 inertia = 1.3333  
Cluster 3 inertia = 0  

$\text{WCSS} = 0.5 + 1.3333 + 0 = 1.8333$


#### **Case 4: Number of Clusters = 4**

- Cluster 1: A, B with centroid $(1.5, 1)$  
- Cluster 2: C, F with centroid $(4, 3.5)$  
- Cluster 3: D with centroid $(5, 4)$  
- Cluster 4: E with centroid $(1, 2)$  

Cluster 1 inertia = 0.5  
Cluster 2 inertia = 0.5  
Cluster 3 inertia = 0  
Cluster 4 inertia = 0  

$\text{WCSS} = 0.5 + 0.5 + 0 + 0 = 1.0$


#### **Case 5: Number of Clusters = 5**

- Cluster 1: A, B → centroid $(1.5, 1)$  
- Cluster 2: F → centroid $(4, 4)$  
- Cluster 3: C → centroid $(4, 3)$  
- Cluster 4: D → centroid $(5, 4)$  
- Cluster 5: E → centroid $(1, 2)$  

Cluster 1 inertia = 0.5  
Cluster 2 inertia = 0  
Cluster 3 inertia = 0  
Cluster 4 inertia = 0  
Cluster 5 inertia = 0  

$\text{WCSS} = 0.5 + 0 + 0 + 0 + 0 = 0.5$

In summary we have from the table below that as $k$ increases, WCSS decreases but the **rate of improvement slows sharply after $k = 2$**. This “elbow” indicates that **$k = 2$** is the optimal number of clusters for this dataset.

| Number of Clusters ($k$) | WCSS |
|:-------------------------:|:----:|
| 1 | 24.3334 |
| 2 | 2.6668 |
| 3 | 1.8333 |
| 4 | 1.0000 |
| 5 | 0.5000 |


The K-Means method has the drawback of being sensitive to the initialization of the centroids or mean points, which makes it difficult to use. A centroid that is initialized as a "far-off" point may end up with no points connected with it, while at the same time more than one cluster may end up associated with a single centroid. Several more centroids may be initialized into the same cluster, resulting in erroneous clustering. This is as a result of the randomness in initializing our centroids.

To overcome the aforementioned drawback, we use K-means++ to attempt to eliminate the randomness in initializing our cluster centers. The cluster centers are better initialized as a result of this method, and the clustering is improved. Except for initialization, the rest of the technique is identical to the standard K-means algorithm. As a consequence, K-means++ combines the traditional K-means approach with improved centroid initialization.

- Take a center $C^{1}$, randomly chosen from the dataset.
- Calculate $D(x)$, the distance between $x$ and the nearest center that has already been picked, for each data point $x$ that has not been chosen yet.
- Using a weighted probability distribution, choose one new data point at random as a new center, with a probability proportional to the distance squared $(D(x)^2)$.
- Repeat Steps 2 and 3 until k centers have been chosen.
