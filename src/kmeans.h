#ifndef _KMEANS_CUDA_H
#define _KMEANS_CUDA_H

void *kmeans_cuda(struct options_t *args,
                  int *numpoints,
                  double **points,
                  double **centers,
                  int **cluster_assignments);

void *kmeans_cuda_thrust(struct options_t *args,
                         int *numpoints,
                         double **points,
                         double **centers,
                         int **cluster_assignments);

void *kmeans_cpu(struct options_t *args,
                 int *numpoints,
                 double **points,
                 double **centers,
                 int **cluster_assignments);

double l2_distance(double *a, double *b, int dim_size);

#endif
