#include <argparse.h>

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <an integer specifying the number of clusters>" << std::endl;
        std::cout << "\t--dims or -d <an integer specifying the dimension of the points>" << std::endl;
        std::cout << "\t--inputfilename or -i <a string specifying the input filename>" << std::endl;
        std::cout << "\t--max_num_iter or -m <an integer specifying the maximum number of iterations>" << std::endl;
        std::cout << "\t--threshold or -t <a double specifying the threshold for convergence test>" << std::endl;
        std::cout << "\t[Optional] -c <a flag to control the output of your program>" << std::endl;
        std::cout << "\t[Optional] --use_shared_mem <a flag to use shared mem when running on gpu>" << std::endl;
        std::cout << "\t[Optional] --use_thrust <a flag to use thrust when running on gpu>" << std::endl;
        std::cout << "\t[Optional] --use_alternate <a flag to use alternate cuda implementation of kmeans that does not use shared mem>" << std::endl;
        std::cout << "\t--seed or -s <an integer specifying the seed for rand()>" << std::endl;
        std::cout << "\t[Optional] --use_cpu <a flag for serial implementation on cpu>" << std::endl;
        exit(0);
    }

    opts->use_cpu = false;
    opts->use_shared_mem = false;
    opts->use_thrust = false;
    opts->use_alternate = false;
    opts->c = false;
    opts->time_data_transfer = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"c", no_argument, NULL, 'c'},
        {"use_shared_mem", no_argument, NULL, 'y'},
        {"use_thrust", no_argument, NULL, 'z'},
        {"use_alternate", no_argument, NULL, 'a'},
        {"seed", required_argument, NULL, 's'},
        {"use_cpu", no_argument, NULL, 'x'},
        {"time_data_transfer", no_argument, NULL, 'b'},
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:cxyzab", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->inputfilename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->c = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'x':
            opts->use_cpu = true;
        case 'y':
            opts->use_shared_mem = true;
            break;
        case 'z':
            opts->use_thrust = true;
            break;
        case 'a':
            opts->use_alternate = true;
            break;
        case 'b':
            opts->time_data_transfer = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
