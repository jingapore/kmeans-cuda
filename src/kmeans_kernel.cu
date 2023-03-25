#include <cuda_runtime.h>
#include <cfloat>
#include "argparse.h"
#include "kmeans.h"
#include <chrono>

__global__ void assign_centroids_shared_mem_kernel(int num_points, int num_clusters, int dim_size, double *d_points, double *d_centers, double *d_new_centers, int *d_new_centers_count, int *d_cluster_assignments)
{
    extern __shared__ char shared_arr[];
    double *shared_centers = (double *)shared_arr;
    double *shared_new_centers = (double *)(num_clusters * dim_size * sizeof(double) + shared_arr);
    int *shared_new_centers_count = (int *)(2 * (num_clusters * dim_size * sizeof(double)) + shared_arr);

    // copy gen mem to shared mem for shared_centers
    // each thread represents a dim in each centroid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int work_per_thread = (num_clusters * dim_size) / blockDim.x;
    int remainder = (num_clusters * dim_size) % blockDim.x;
    if (threadIdx.x < remainder)
    {
        work_per_thread += 1;
    }
    for (int work_id = 0; work_id < work_per_thread; ++work_id)
    {
        int cluster_dim_id = threadIdx.x + work_id * blockDim.x;
        // printf("cluster_dim_id of thread %d: %d\n", threadIdx.x, cluster_dim_id);
        if (cluster_dim_id < num_clusters * dim_size)
        {
            shared_centers[cluster_dim_id] = d_centers[cluster_dim_id];
            shared_new_centers[cluster_dim_id] = 0; // conveniently assign to 0
            if (cluster_dim_id < num_clusters)
            {
                shared_new_centers_count[cluster_dim_id] = 0; // conveniently assign to 0
            }
        }
    }
    __syncthreads();

    // assign centroids, reading from shared mem
    // write to shared mem (a) new centers and (b) new centers count, with atomicAdd
    // each thread represents a data point
    if (tid < num_points)
    {
        int closest_cluster = -1;
        double min_distance = DBL_MAX;

        // Find the closest cluster centroid
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double distance = 0.0f;
            for (int dim = 0; dim < dim_size; ++dim)
            {
                double diff = d_points[tid * dim_size + dim] - shared_centers[cluster_id * dim_size + dim];
                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_cluster = cluster_id;
            }
        }

        // Update cluster assignment
        d_cluster_assignments[tid] = closest_cluster;
        atomicAdd(&(shared_new_centers_count[closest_cluster]), 1);
        // printf("closest centroid for tid %d is: %d\n", tid, closest_cluster);

        for (int dim = 0; dim < dim_size; ++dim)
        {
            // atomicAdd for double requires us to compile with `-arch=sm_60`
            atomicAdd(&(shared_new_centers[closest_cluster * dim_size + dim]), d_points[tid * dim_size + dim]);
        }
    }

    __syncthreads();

    // copy out shared mem for (a) new centers, (b) new centers count.
    // after this step, new_centers in gen mem would have accumulated all the new points across blocks
    // but we'll have to divide it by the sum accumulated in new centers,
    // and this will be achieved `update_centroids_shared_mem_kernel`.
    for (int work_id = 0; work_id < work_per_thread; ++work_id)
    {
        int cluster_dim_id = threadIdx.x + work_id * blockDim.x;
        if (cluster_dim_id < num_clusters * dim_size)
        {
            atomicAdd(&(d_new_centers[cluster_dim_id]), shared_new_centers[cluster_dim_id]);
            if (cluster_dim_id < num_clusters)
            {
                atomicAdd(&(d_new_centers_count[cluster_dim_id]), shared_new_centers_count[cluster_dim_id]); // conveniently copy out cluster count to global mem
            }
        }
    }
}

__global__ void update_centroids_shared_mem_kernel(int num_clusters, int dim_size, double *d_centers, double *d_new_centers, int *d_new_centers_count)
{
    // each thread represents a dim in each centroid.
    // update d_centers with new centroid from (a) new centers and (b) new centers count.
    // (a) and (b) above are in gen mem, no point reading them into shared mem.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster_id = tid / dim_size;

    if (tid < num_clusters * dim_size)
    {

        // Avoid division by zero
        if (d_new_centers_count[cluster_id] > 0)
        {
            // printf("updating cluster id %d by tid %d where d_new_centers is %.3f and new cluster count is %d \n", cluster_id, tid, d_new_centers[tid], d_new_centers_count[cluster_id]);
            d_centers[tid] = d_new_centers[tid] / d_new_centers_count[cluster_id];
        }
    }
}

__global__ void assign_centroids_kernel(int num_points, int num_clusters, int dim_size, double *d_points, double *d_centers, double *d_new_centers, int *d_new_centers_count, int *d_cluster_assignments)
{
    // get thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points)
    {
        int closest_cluster = -1;
        double min_distance = DBL_MAX;

        // Find the closest cluster centroid
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double distance = 0.0f;
            for (int dim = 0; dim < dim_size; ++dim)
            {
                double diff = d_points[tid * dim_size + dim] - d_centers[cluster_id * dim_size + dim];
                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_cluster = cluster_id;
            }
        }

        // Update cluster assignment
        d_cluster_assignments[tid] = closest_cluster;
        atomicAdd(&(d_new_centers_count[closest_cluster]), 1);
        for (int dim = 0; dim < dim_size; ++dim)
        {
            // atomicAdd for double requires us to compile with `-arch=sm_60`
            atomicAdd(&(d_new_centers[closest_cluster * dim_size + dim]), d_points[tid * dim_size + dim]);
        }
    }
}

__global__ void assign_centroids_kernel_alternate(int num_points, int num_clusters, int dim_size, double *d_points, double *d_centers, int *d_cluster_assignments)
{
    // get thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points)
    {
        int closest_cluster = -1;
        double min_distance = DBL_MAX;

        // Find the closest cluster centroid
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double distance = 0.0f;
            for (int dim = 0; dim < dim_size; ++dim)
            {
                double diff = d_points[tid * dim_size + dim] - d_centers[cluster_id * dim_size + dim];
                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_cluster = cluster_id;
            }
        }

        // Update cluster assignment
        d_cluster_assignments[tid] = closest_cluster;
    }
}

__global__ void update_centroids_kernel_alternate(int num_points, int num_clusters, int dim_size, double *d_points, double *d_centers, int *d_cluster_assignments)
{
    // no need for atomicAdd, since each thread is updating its own centroid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster_id = tid / dim_size;
    int dim = tid % dim_size;

    if (tid < num_clusters * dim_size)
    {
        double sum = 0.0;
        int count = 0;

        for (int point_id = 0; point_id < num_points; ++point_id)
        {
            if (d_cluster_assignments[point_id] == cluster_id)
            {
                sum += d_points[point_id * dim_size + dim];
                count++;
            }
        }

        // Avoid division by zero
        if (count > 0)
        {
            d_centers[tid] = sum / count;
        }
    }
}

#define CUDA_CHECK_ERROR(err)                                                              \
    if (err != cudaSuccess)                                                                \
    {                                                                                      \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(-1);                                                                          \
    }

void *kmeans_cuda(struct options_t *args,
                  int *numpoints,
                  double **points,
                  double **centers,
                  int **cluster_assignments)
{
    int num_points = *numpoints;
    bool use_shared_mem = args->use_shared_mem;
    bool use_alternate = args->use_alternate;
    bool time_data_transfer = args->time_data_transfer;
    int num_clusters = args->num_clusters;
    int dim_size = args->dims;
    int max_num_iter = args->max_num_iter;
    double convergence_threshold = args->threshold;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float data_transfer_ms = 0.0;
    float e2e_duration = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now(); // this times e2e duration, but we only output it if time_data_transfer ise set

    // copy data to device
    double *d_points;
    double *d_centers;
    int *d_cluster_assignments;
    double *d_new_centers;    // not used in suboptimal alternative implementation
    int *d_new_centers_count; // not used in suboptimal alternative implementation
    cudaMalloc((void **)&d_points, num_points * dim_size * sizeof(double));
    cudaMalloc((void **)&d_centers, num_clusters * dim_size * sizeof(double));
    cudaMalloc((void **)&d_cluster_assignments, num_points * sizeof(int));
    cudaMalloc((void **)&d_new_centers, num_clusters * dim_size * sizeof(double));
    cudaMalloc((void **)&d_new_centers_count, num_clusters * sizeof(int));

    if (time_data_transfer)
    {
        cudaEventRecord(start);
    }
    cudaMemcpy(d_points, *points, num_points * dim_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, *centers, num_clusters * dim_size * sizeof(double), cudaMemcpyHostToDevice);

    if (time_data_transfer)
    {
        cudaEventRecord(stop);
        float duration = 0;
        cudaEventElapsedTime(&duration, start, stop);
        data_transfer_ms += duration;
    }

    int threads_per_block = 512; // can soft code this and see how it changes
    int num_blocks_assignment = (num_points + threads_per_block - 1) / threads_per_block;
    int num_blocks_update_centroids = (num_clusters * dim_size + threads_per_block - 1) / threads_per_block;
    float delta_ms = 0.0;
    double *h_previous_centers = (double *)malloc(num_clusters * dim_size * sizeof(double));
    double *h_current_centers = (double *)malloc(num_clusters * dim_size * sizeof(double));

    int iter_to_converge = 0;
    float total_time_taken = 0.0;
    for (int iter = 0; iter < max_num_iter; ++iter)
    {
        iter_to_converge += 1;
        // Copy the current centroids to the host
        if (time_data_transfer)
        {
            cudaEventRecord(start);
        }
        cudaMemcpy(h_previous_centers, d_centers, num_clusters * dim_size * sizeof(double), cudaMemcpyDeviceToHost);
        if (time_data_transfer)
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float duration = 0;
            cudaEventElapsedTime(&duration, start, stop);
            data_transfer_ms += duration;
        }
        if (use_alternate)
        {
            // this takes (dim * num_clusters + n_points) time instead of (dim * num_clusters) time
            // it should only make sense if (dim * num_clusters) dominate the number of points
            // it isn't an optimal implementation, but we keep it here for the sake of experimentation
            cudaEventRecord(start);
            assign_centroids_kernel_alternate<<<num_blocks_assignment, threads_per_block>>>(num_points, num_clusters, dim_size, d_points, d_centers, d_cluster_assignments);
            cudaDeviceSynchronize();

            update_centroids_kernel_alternate<<<num_blocks_update_centroids, threads_per_block>>>(num_points, num_clusters, dim_size, d_points, d_centers, d_cluster_assignments);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&delta_ms, start, stop);
            total_time_taken += delta_ms;
        }
        else if (use_shared_mem)
        {
            cudaEventRecord(start);
            // kernel to (a) assign centroid and (b) calculate block-level centroid assignments
            assign_centroids_shared_mem_kernel<<<num_blocks_assignment, threads_per_block, 2 * (num_clusters * dim_size * sizeof(double)) + (num_clusters * sizeof(int))>>>(num_points,
                                                                                                                                                                            num_clusters,
                                                                                                                                                                            dim_size,
                                                                                                                                                                            d_points,
                                                                                                                                                                            d_centers,
                                                                                                                                                                            d_new_centers,
                                                                                                                                                                            d_new_centers_count,
                                                                                                                                                                            d_cluster_assignments);
            cudaDeviceSynchronize();

            // kernel to calculate grid-level centroid assignments
            // we require new kernel, because we need all blocks to be in sync and that can only be achieved with kernels
            update_centroids_shared_mem_kernel<<<num_blocks_update_centroids, threads_per_block>>>(num_clusters, dim_size, d_centers, d_new_centers, d_new_centers_count);
            cudaDeviceSynchronize();
            cudaMemset(d_new_centers_count, 0, num_clusters * sizeof(int));         // set to zero
            cudaMemset(d_new_centers, 0, num_clusters * dim_size * sizeof(double)); // set to zero
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&delta_ms, start, stop);
            total_time_taken += delta_ms;
        }
        else
        {
            // standard approach
            cudaEventRecord(start);
            assign_centroids_kernel<<<num_blocks_assignment, threads_per_block>>>(num_points, num_clusters, dim_size, d_points, d_centers, d_new_centers, d_new_centers_count, d_cluster_assignments);
            cudaDeviceSynchronize();
            update_centroids_shared_mem_kernel<<<num_blocks_update_centroids, threads_per_block>>>(num_clusters, dim_size, d_centers, d_new_centers, d_new_centers_count);
            cudaDeviceSynchronize();
            cudaMemset(d_new_centers_count, 0, num_clusters * sizeof(int));         // set to zero
            cudaMemset(d_new_centers, 0, num_clusters * dim_size * sizeof(double)); // set to zero
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&delta_ms, start, stop);
            total_time_taken += delta_ms;
        }
        // Check for convergence by comparing the change in centroids
        if (time_data_transfer)
        {
            cudaEventRecord(start);
        }
        cudaMemcpy(h_current_centers, d_centers, num_clusters * dim_size * sizeof(double), cudaMemcpyDeviceToHost);
        if (time_data_transfer)
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float duration = 0;
            cudaEventElapsedTime(&duration, start, stop);
            data_transfer_ms += duration;
        }
        double max_centroid_change = 0.0;
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double centroid_change = l2_distance(&h_previous_centers[cluster_id * dim_size], &h_current_centers[cluster_id * dim_size], dim_size);
            max_centroid_change = fmax(max_centroid_change, centroid_change);
        }

        // Check if the maximum centroid change is below the threshold
        if (max_centroid_change < convergence_threshold)
        {
            // printf("Converged (max centroid change=%.6f)\n", max_centroid_change);
            break; // Exit the loop if converged
        }
    }
    float time_per_iter_in_ms = total_time_taken / iter_to_converge;
    printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
    // Copy results back to host
    if (time_data_transfer)
    {
        cudaEventRecord(start);
    }
    cudaMemcpy(*cluster_assignments, d_cluster_assignments, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*centers, d_centers, num_clusters * dim_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (time_data_transfer)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float duration = 0;
        cudaEventElapsedTime(&duration, start, stop);
        data_transfer_ms += duration;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    if (time_data_transfer)
    {
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        e2e_duration += elapsed_time.count();
        e2e_duration = e2e_duration / 1000; // 1000 microseconds in a millisecond
        printf("data transfer time is: %.3fms\n", data_transfer_ms);
        printf("e2e duration is: %.3fms\n", e2e_duration);
    }
    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_cluster_assignments);
    cudaFree(d_new_centers);
    cudaFree(d_new_centers_count);

    return NULL; // You can return additional information or status if needed
}