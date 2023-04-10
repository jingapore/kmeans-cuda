# 1. TLDR

We examine the theoretical and empirical boost in speed, when we move from sequential implementation of K-means on the CPU to parallel implementations on the GPU. Theoretically, we can expect a speed up of over 1000x. Empirically, we observe a ~10x speed up. When studying this divergence between expectation and reality, we account for factors: namely data transfer and synchronisation “costs”. We conclude with remarks on the use of higher-level frameworks like Thrust and possible speed ups from using libraries like GEMM.

# 2. CUDA Implementation of Parallel K-means
## 2.1 Environment
## 2.2 Approach

<p align="center"><b>fig 1: shared memory for computing closest centroid</b></p>
<p align="center">
    <img width="66%" src="img/fig_1.png">
</p>

# 3. Speedup: Expectation vs Reality
## 3.1 Theoretical Estimate
## 3.2 Empirical Results

<p align="center"><b>fig 2: timing of seq vs parallel implementation</b></p>
<p align="center">
    <img width="66%" src="img/fig_2.png">
</p>

<p align="center"><b>fig 3: iterations to convergence at threshold of 10^-6</b></p>
<p align="center">
    <img width="66%" src="img/fig_3.png">
</p>

## 3.3 Analysis: Why Reality was Different from Expectation

# 4. Parting Remarks