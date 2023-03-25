#include <io.h>
#include <iostream>
#include <sstream>

void read_file(struct options_t *args,
			   int *numpoints,
			   double **points,
			   double **centers)
{

	// Open file
	std::ifstream in;
	in.open(args->inputfilename);
	// Get num points in input file, this is from the first line
	in >> *numpoints;

	// allocate memory, flattening to array
	*points = (double *)malloc(*numpoints * args->dims * sizeof(double));
	*centers = (double *)malloc(args->num_clusters * args->dims * sizeof(double));

	// read input vals while concurrently allocating memory for each row-col
	std::string line;
	int line_number = 0;
	int num_double_elements = 0;
	double double_element;
	while (std::getline(in, line))
	{
		if (line_number > 0)
		{
			std::istringstream iss(line);
			int col_index = 0;
			while (iss >> double_element)
			{
				// printf("double_element is %f\n", double_element);
				if (col_index > 0)
				{
					// if col_index==0, it is just row index in input file
					(*points)[(line_number - 1) * args->dims + col_index - 1] = double_element;
				}
				col_index++;
			}
			if (col_index - 1 != args->dims)
			{
				std::cerr << "Warning: Number of values read does not match expected dimensions." << std::endl;
			}
		}
		line_number++;
	}

	// TODO: check numpoints and linenumber are the same
	// also check col_index and num_double_elements are the same
}

// void write_file(struct options_t *args,
// 				struct prefix_sum_args_t *opts)
// {
// 	// Open file
// 	std::ofstream out;
// 	out.open(args->out_file, std::ofstream::trunc);

// 	// Write solution to output file
// 	for (int i = 0; i < opts->n_vals; ++i)
// 	{
// 		out << opts->output_vals[i] << std::endl;
// 	}

// 	out.flush();
// 	out.close();

// 	// Free memory
// 	free(opts->input_vals);
// 	free(opts->output_vals);
// }
