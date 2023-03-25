#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int num_clusters;
    int dims;
    char *inputfilename;
    int max_num_iter;
    double threshold;
    bool c;
    bool use_alternate;
    bool use_shared_mem;
    bool use_thrust;
    bool time_data_transfer;
    int seed;
    bool use_cpu;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
