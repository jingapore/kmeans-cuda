#include "argparse.h"
#include "io.h"
#include "kmeans.h"

using namespace std;

static unsigned long int kmeans_next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand()
{
    kmeans_next = kmeans_next * 1103515245 + 12345;
    return (unsigned int)(kmeans_next / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed)
{
    kmeans_next = seed;
}

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // read in data from file
    int _numpoints;
    double *points;
    double *centers;
    int *cluster_assignments = (int *)malloc(_numpoints * opts.dims * sizeof(int));
    int *center_counter = (int *)malloc(opts.num_clusters * sizeof(int)); // we will increment this, which will be useful for recalculating centroids
    read_file(&opts, &_numpoints, &points, &centers);

    kmeans_srand(opts.seed);
    for (int i = 0; i < opts.num_clusters; i++)
    {
        int index = kmeans_rand() % _numpoints;
        for (int j = 0; j < opts.dims; j++)
        {
            centers[i * opts.dims + j] = points[index * opts.dims + j];
        }
    }

    if (opts.use_cpu)
    {
        // printf("starting sequential implementation\n");
        kmeans_cpu(&opts, &_numpoints, &points, &centers, &cluster_assignments);
    }
    else if (opts.use_thrust)
    {
        // printf("starting thrust implementation\n");
        kmeans_cuda_thrust(&opts, &_numpoints, &points, &centers, &cluster_assignments);
        // for (int i = 0; i < _numpoints; ++i)
        // {
        //     printf("Point %d belongs to cluster %d\n", i, cluster_assignments[i]);
        // }
    }
    else
    {
        // printf("starting cuda implementation\n");
        kmeans_cuda(&opts, &_numpoints, &points, &centers, &cluster_assignments);
    }

    if (opts.c)
    {
        // output centroids of clusters
        for (int clusterId = 0; clusterId < opts.num_clusters; clusterId++)
        {
            printf("%d ", clusterId);
            for (int d = 0; d < opts.dims; d++)
                printf("%lf ", centers[clusterId * opts.dims + d]);
            printf("\n");
        }
    }
    else
    {
        // output centroid ids for each point
        printf("clusters:");
        for (int p = 0; p < _numpoints; p++)
        {
            printf(" %d", cluster_assignments[p]);
        }
    }
}
