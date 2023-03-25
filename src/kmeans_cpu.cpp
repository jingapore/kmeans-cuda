#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include "kmeans.h"
#include "argparse.h"

double l2_distance(double *a, double *b, int dim_size)
{
    double sum = 0.0;
    for (int dim = 0; dim < dim_size; ++dim)
    {
        double diff = a[dim] - b[dim];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int find_closest_centroid(double *point, double *centroids, int num_clusters, int dim_size)
{
    int closest_cluster = -1;
    double min_distance = DBL_MAX;

    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        double distance_to_centroid = l2_distance(point, &centroids[cluster_id * dim_size], dim_size);
        if (distance_to_centroid < min_distance)
        {
            min_distance = distance_to_centroid;
            closest_cluster = cluster_id;
        }
    }
    return closest_cluster;
}

void *kmeans_cpu(struct options_t *args,
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
    double *centers_previous = (double *)malloc(num_clusters * dim_size * sizeof(double));
    int *center_count = (int *)malloc(num_clusters * sizeof(int)); // reset to zero
                                                                   // Main loop for k-means iterations
    int iter_to_converge = 0;
    double total_time_taken = 0.0;
    for (int iter = 0; iter < max_num_iter; ++iter)
    {
        iter_to_converge += 1;
        memset(center_count, 0, num_clusters * sizeof(int));
        // copy out centroids, as we need it to calculate convergence later
        for (int i = 0; i < num_clusters * dim_size; ++i)
        {
            centers_previous[i] = (*centers)[i];
            (*centers)[i] = 0.0; // set to zero as we will update this while finding closest centroid
        }

        // Assign each point to the nearest cluster
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int point_id = 0; point_id < num_points; ++point_id)
        {
            int closest_cluster = find_closest_centroid(&((*points)[point_id * dim_size]), centers_previous, num_clusters, dim_size);
            (*cluster_assignments)[point_id] = closest_cluster;
            (center_count)[closest_cluster]++;
            for (int dim = 0; dim < dim_size; ++dim)
            {
                (*centers)[closest_cluster * dim_size + dim] += (*points)[point_id * dim_size + dim]; // we update centers here, which is why we need centers_previous
            }
        }

        // Update cluster centers by taking the mean of assigned points
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            if (center_count[cluster_id] > 0)
            {
                for (int dim = 0; dim < dim_size; ++dim)
                {
                    (*centers)[cluster_id * dim_size + dim] /= center_count[cluster_id];
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        // std::cout << "Iteration " << iter << " took " << elapsed_time.count() << " ms" << std::endl;
        total_time_taken += elapsed_time.count();

        // Check for convergence by comparing the change in cluster centers
        double max_centroid_change = 0.0;
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            double centroid_change = l2_distance(&((*centers)[cluster_id * dim_size]), &centers_previous[cluster_id * dim_size], dim_size);
            max_centroid_change = fmax(max_centroid_change, centroid_change);
        }

        // Check if the maximum centroid change is below the threshold
        if (max_centroid_change < convergence_threshold)
        {
            // printf("Converged after %d iterations (max centroid change=%.6f)\n", iter, max_centroid_change);
            break; // Exit the loop if converged
        }
    }
    double time_per_iter_in_ms = total_time_taken / (iter_to_converge * 1000 * 1000);
    printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);

    free(center_count);
    free(centers_previous);

    return NULL; // You can return additional information or status if needed
}