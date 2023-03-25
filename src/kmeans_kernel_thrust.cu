#include <cuda_runtime.h>
#include <cfloat>
#include "argparse.h"
#include "kmeans.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

struct SquaredDistance : public thrust::unary_function<int, void>
{
    int dim_size;
    int num_clusters;
    thrust::device_ptr<double> d_points;
    thrust::device_ptr<double> d_centers;
    int *d_cluster_assignments;

    SquaredDistance(int dim_size, int num_clusters, thrust::device_ptr<double> d_points, thrust::device_ptr<double> d_centers, int *d_cluster_assignments)
        : dim_size(dim_size), num_clusters(num_clusters), d_points(d_points), d_centers(d_centers), d_cluster_assignments(d_cluster_assignments) {}

    __host__ __device__ void operator()(int idx)
    {
        double min_distance = DBL_MAX;
        int nearest_centroid_index = -1;

        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx)
        {
            double distance = 0;
            for (int i = 0; i < dim_size; ++i)
            {
                double diff = d_points[idx * dim_size + i] - d_centers[cluster_idx * dim_size + i];
                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                nearest_centroid_index = cluster_idx;
            }
        }

        d_cluster_assignments[idx] = nearest_centroid_index;
    }
};

struct ClusterSumsAverage : public thrust::unary_function<thrust::tuple<int, double>, void>
{
    int dim_size;
    int *cluster_counts;
    thrust::device_ptr<double> d_centers;

    ClusterSumsAverage(int dim_size, int *cluster_counts, thrust::device_ptr<double> d_centers)
        : dim_size(dim_size), cluster_counts(cluster_counts), d_centers(d_centers) {}

    __host__ __device__ void operator()(const thrust::tuple<int, double> &centroid_dim_idx_and_centroid_accumulated_sum)
    {
        // get centroid_idx from centroid_dim_idx
        int centroid_dim_idx = thrust::get<0>(centroid_dim_idx_and_centroid_accumulated_sum);
        int centroid_idx = centroid_dim_idx / dim_size;
        // extract centroid count
        int count = cluster_counts[centroid_idx];
        // divide accumulated sum by centroid_count
        // printf("centroid_idx is: %d, centroid_dim_idx is %d, count is %d, sum is %.05f\n", centroid_idx, centroid_dim_idx, count, thrust::get<1>(centroid_dim_idx_and_centroid_accumulated_sum));
        if (count > 0)
        {
            d_centers[centroid_dim_idx] = thrust::get<1>(centroid_dim_idx_and_centroid_accumulated_sum) / count;
        }
    }
};

struct ClusterDimExpansion : public thrust::unary_function<thrust::tuple<int, int>, void>
{
    int dim_size;
    int *d_cluster_assignments_with_dims;

    ClusterDimExpansion(int dim_size, int *d_cluster_assignments_with_dims)
        : dim_size(dim_size), d_cluster_assignments_with_dims(d_cluster_assignments_with_dims) {}

    __host__ __device__ void operator()(const thrust::tuple<int, int> &point_cluster_assignment)
    {

        for (int dim_idx = 0; dim_idx < dim_size; ++dim_idx)
        {
            int idx_new = (thrust::get<1>(point_cluster_assignment) * dim_size) + dim_idx;
            d_cluster_assignments_with_dims[thrust::get<0>(point_cluster_assignment) * dim_size + dim_idx] = idx_new;
        }
    }
};

void *kmeans_cuda_thrust(struct options_t *args,
                         int *numpoints,
                         double **points,
                         double **centers,
                         int **cluster_assignments)
{
    int num_points = *numpoints;
    int num_clusters = args->num_clusters;
    int dim_size = args->dims;
    int max_num_iter = args->max_num_iter;
    double convergence_threshold = args->threshold;

    thrust::host_vector<double> h_points(*points, *points + num_points * dim_size);
    thrust::host_vector<double> h_centers(*centers, *centers + num_clusters * dim_size);
    thrust::host_vector<double> h_previous_centers(*centers, *centers + num_clusters * dim_size);
    thrust::host_vector<int> h_cluster_assignments(*cluster_assignments, *cluster_assignments + num_points);

    thrust::device_vector<double> d_points = h_points;
    thrust::device_vector<double> d_centers = h_centers;
    thrust::device_vector<int> d_cluster_assignments = h_cluster_assignments;
    thrust::device_vector<int> d_cluster_assignments_with_dims(num_points * dim_size); // used for reduce operations

    // double *d_points_ptr = thrust::raw_pointer_cast(d_points.data());
    // double *d_centers_ptr = thrust::raw_pointer_cast(d_centers.data());
    thrust::device_ptr<double> d_points_ptr = thrust::device_pointer_cast(d_points.data());
    thrust::device_ptr<double> d_centers_ptr = thrust::device_pointer_cast(d_centers.data());
    int *d_cluster_assignments_ptr = thrust::raw_pointer_cast(d_cluster_assignments.data());
    int *d_cluster_assignments_with_dims_ptr = thrust::raw_pointer_cast(d_cluster_assignments_with_dims.data());

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float delta_ms = 0.0;
    int iter_to_converge = 0;
    float total_time_taken = 0.0;
    for (int iter = 0; iter < max_num_iter; ++iter)
    {
        iter_to_converge += 1;
        h_previous_centers = h_centers;
        d_centers = h_centers;
        d_points = h_points;
        d_points_ptr = thrust::device_pointer_cast(d_points.data());
        d_centers_ptr = thrust::device_pointer_cast(d_centers.data());
        cudaEventRecord(start);
        // get cluster assignments
        SquaredDistance distance_calculator(dim_size, num_clusters, d_points_ptr, d_centers_ptr, d_cluster_assignments_ptr);
        thrust::counting_iterator<int> idx(0);
        thrust::for_each(idx, idx + num_points, distance_calculator);
        h_cluster_assignments = d_cluster_assignments; // copy out before sorting, else cluster assignment becomes jumbled

        // after getting cluster assignments, calculate new centroids
        // reduce 1: sum
        // reduce 2: get counts
        // average = summation / counts

        // to store reduce 1
        thrust::device_vector<double> cluster_sums(num_clusters * dim_size);
        thrust::device_vector<int> cluster_sums_keys(num_clusters * dim_size);

        // to store reduce 2
        thrust::device_vector<int> cluster_counts(num_clusters, 0);
        thrust::device_vector<int> cluster_counts_keys(num_clusters);
        thrust::device_vector<int> cluster_counts_expanded(num_clusters * dim_size);

        // reduce 1: sum
        // expand d_cluster_assignments to keys that reflect dimensions
        ClusterDimExpansion cluster_dim_expansion(dim_size, d_cluster_assignments_with_dims_ptr);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(idx, d_cluster_assignments.begin())), thrust::make_zip_iterator(thrust::make_tuple(idx + num_points, d_cluster_assignments.end())), cluster_dim_expansion);
        thrust::stable_sort_by_key(d_cluster_assignments_with_dims.begin(),
                                   d_cluster_assignments_with_dims.end(),
                                   d_points.begin());
        thrust::reduce_by_key(d_cluster_assignments_with_dims.begin(),
                              d_cluster_assignments_with_dims.end(),
                              d_points.begin(),
                              cluster_sums_keys.begin(),
                              cluster_sums.begin());

        // reduce 2: get counts
        thrust::stable_sort_by_key(d_cluster_assignments.begin(),
                                   d_cluster_assignments.end(),
                                   thrust::make_discard_iterator());
        thrust::reduce_by_key(d_cluster_assignments.begin(),
                              d_cluster_assignments.end(),
                              thrust::constant_iterator<int>(1),
                              cluster_counts_keys.begin(),
                              cluster_counts.begin());

        // average
        int *cluster_counts_ptr = thrust::raw_pointer_cast(cluster_counts.data());
        ClusterSumsAverage cluster_sums_average(dim_size, cluster_counts_ptr, d_centers_ptr);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(idx, cluster_sums.begin())), thrust::make_zip_iterator(thrust::make_tuple(idx + num_clusters * dim_size, cluster_sums.end())), cluster_sums_average);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&delta_ms, start, stop);
        total_time_taken += delta_ms;
        // printf("Iteration %d took %.5f ms\n", iter, delta_ms);

        // check convergence
        h_centers = d_centers; // copy out, else the next loop will malfunction, possibly due to device mem being reallocated by thrust
        double max_centroid_change = 0.0;
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double centroid_change = l2_distance(&h_previous_centers[cluster_id * dim_size], &h_centers[cluster_id * dim_size], dim_size);
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
    thrust::copy(h_centers.begin(), h_centers.end(), *centers);
    thrust::copy(h_cluster_assignments.begin(), h_cluster_assignments.end(), *cluster_assignments);

    return NULL;
}