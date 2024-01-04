# 1. TLDR

We examine the theoretical and empirical boost in speed, when we move from sequential implementation of K-means on the CPU to parallel implementations on the GPU. Theoretically, we can expect a speed up of over 1000x. Empirically, we observe a ~10x speed up. When studying this divergence between expectation and reality, we account for factors: namely data transfer and synchronisation “costs”. We conclude with remarks on the use of higher-level frameworks like Thrust and possible speed ups from using libraries like GEMM.

# 2. CUDA Implementation of Parallel K-means
## 2.1 Environment
Our experiment takes place on a single machine running NVIDA Tesla T4.
## 2.2 Approach
K-means comprises two key steps:
- <u>Stage 1</u>: Compute closest centroid for each point; and
- <u>Stage 2</u>: Update centroids.

In the rest of this report, we use “n_” to denote the following.
- n_points: Number of points in dataset, e.g. 2048, 16384, 65536.
- n_dims: Number of dimensions of points, e.g. 16, 24, 32.
- n_clusters: Number of K-means clusters, e.g. 16.

Our sequential implementation adopts an optimisation trick. We use information gleaned from Stage 1 to speed up Stage 2. We obtain such information at Stage 1 by (a) accumulating the sum of coordinates for each new centroid, and (b) counting points assigned to each new centroid. With such information already gleaned from Stage 1, Stage 2 becomes a more trivial step. At Stage 2, we just average out the summed coordinates of new points in each centroid.

Our parallel implementations are:
- CUDA Basic;
- CUDA with shared memory (“CUDA SharedMem”); and
- CUDA Thrust.

Both CUDA Basic and CUDA SharedMem adopt the same algorithm.
- <u>Stage 1</u>: We launch **n_points** number of threads, and each thread computes the distance to every centroid, and assigns the point to one of the centroids. Each thread also updates the accumulated sum and count of threads, which is used in Stage 2.
- <u>Stage 2</u>: We launch **n_clusters** * **n_dims** number of threads. Each thread divides the accumulated sum of the points newly assigned to a centroids, with the count of the number of points assigned to the new centroids.

CUDA Basic and CUDA SharedMem differ in their memory management approach. <u>Fig 1</u> illustrates how CUDA SharedMem attempts to cut latency by shifting centroid data from the GPU’s general memory to shared memory, although it incurs an overhead of loading centroid data into shared memory for each iteration.

<p align="center"><b>fig 1: shared memory for computing closest centroid</b></p>
<p align="center">
    <img width="66%" src="img/fig_1.png">
</p>

We do not perform the optimisation in Fig 1 for Stage 2, since each thread (where there are **n_clusters** * **n_dims** number of threads) just reads once from general memory. Hence the overhead of loading data into shared memory will not be worth it.

# 3. Speedup: Expectation vs Reality
## 3.1 Theoretical Estimate
The following table does a rough accounting of time required for sequential K-means.

| S/N | Step                                                      | Stage                                      | Estimated Time Required                                  |
|-----|-----------------------------------------------------------|--------------------------------------------|----------------------------------------------------------|
| 1   | Calculate L2 distance                                     | 1, i.e. compute closest centroid for each point | n_points * (n_cluster * n_dims + n_clusters), from calculating distance to all clusters and then getting minimum distance. |
| 2   | Find minimum centroid                                     | 1, i.e. compute closest centroid for each point | n_points * n_clusters                                    |
| 3   | Accumulate point coordinates in preparation for centroid update | 1, i.e. compute closest centroid for each point | n_points * n_clusters * n_dims                           |
| 4   | Update centroid by averaging accumulated point coordinates | 2, i.e. update centroid                      | n_clusters * n_dims                                      |

We use Amdahl’s Law to estimate the speed up, taking into account that our parallel approach spawns  **n_points** number of threads in Stage 1 and **n_clusters** * **n_dims number** of threads in Stage 2. The remaining work that is done sequentially is whatever is not struck out in following table.

| S/N | Step                                                      | Stage                                      | Estimated Time Required                                  |
|-----|-----------------------------------------------------------|--------------------------------------------|----------------------------------------------------------|
| 1   | Calculate L2 distance                                     | 1, i.e. compute closest centroid for each point | ~~n_points *~~ (n_cluster * n_dims + n_clusters), from calculating distance to all clusters and then getting minimum distance. |
| 2   | Find minimum centroid                                     | 1, i.e. compute closest centroid for each point | ~~n_points *~~ n_clusters                                    |
| 3   | Accumulate point coordinates in preparation for centroid update | 1, i.e. compute closest centroid for each point | ~~n_points~~ * n_clusters * n_dims                           |
| 4   | Update centroid by averaging accumulated point coordinates | 2, i.e. update centroid                      | ~~n_clusters * n_dims~~                                      |

The gains from parallelism look very promising, since a large portion of the algorithm’s runtime is dominated by n_points and this can be parallelised. A back-of-envelope calculation estimates that speed-up should be >**1000x**.

## 3.2 Empirical Results

With a convergence threshold of 10-6, we see a speed up of at most 10.44x (Sequential v.s. CUDA Basic where n_points=65536). This is short of our theoretical expectation.

<p align="center"><b>fig 2: timing of seq vs parallel implementation</b></p>
<p align="center">
    <img width="66%" src="img/fig_2.png">
</p>

For completeness, the number of iterations-to-convergence is as follows.

<p align="center"><b>fig 3: iterations to convergence at threshold of 10^-6</b></p>
<p align="center">
    <img width="66%" src="img/fig_3.png">
</p>

## 3.3 Analysis: Why Reality was Different from Expectation
### 3.3.1 Data transfer from host to GPU and <u>within</u> GPU
The first angle that we will explore is data transfer. The broad hypothesis is that the transfer of data from host to GPU has created significant overheads, and this decreases **p** in Amdahl's law, the proportion of the overall runtime that can be parallelised.

However, when comparing CUDA Basic’s end-to-end runtime with the total data transfer “costs”, they do not make up a large proportion. So we discount this hypothesis.

| Dataset       | CUDA Basic end-to-end time (ms) | CUDA Basic cumulative data transfer time (ms) | Proportion of end-to-end time spent on data transfer |
|---------------|---------------------------------|-----------------------------------------------|------------------------------------------------------|
| n2048-d16-c16 | 6.874                           | 0.346                                         | 5.03%                                                |
| n16384-d24-c16| 19.487                          | 0.769                                         | 3.95%                                                |
| n65536-d32-c16| 336.21                          | 2.045                                         | 0.61%                                                |

Also still on the topic of data transfer, we would like to make a brief comment concerning data transfer <u>within</u> the GPU. This relates to our secondary observation, that CUDA SharedMem showed <u>no</u> appreciable speed up over CUDA Basic.

As in <u>Fig 1</u>, the speed up comes from reading from shared memory (step 2 in red), whereas the cost is reading and writing centroids data from general memory to shared memory (step 1 in blue and step 3 in orange). For the test parameters given to K-means, the pros-and-cons cancel each other out. However, if the dimensionality of the centroids increases and/or the number of centroids grows, CUDA SharedMem may outperform CUDA Basic since the one-off cost of transferring data to shared memory per iteration will be offset by the latency shaved off from the read-writes concerning centroid data.

### 3.3.2 Synchronisation Costs


# 4. Parting Remarks