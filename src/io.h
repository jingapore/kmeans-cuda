#ifndef _IO_H
#define _IO_H

#include <argparse.h>
#include <iostream>
#include <fstream>

void read_file(struct options_t *args,
               int *numpoints,
               double **points,
               double **centers);

void write_file(struct options_t *args,
                struct prefix_sum_args_t *opts);

#endif
