import argparse

def add_default_args(parser):

    parser.add_argument('--topo_name', type=str, default = 'Facebook_tor_a')
    parser.add_argument('--paths_file', type=str, default = 'tunnels.txt')
    parser.add_argument('--path_num', type=int, default=3)
    
    parser.add_argument('--hist_len', type=int, default = 12)

    parser.add_argument('--TE_solver', type=str, default = 'Jupiter')

    # Jupiter
    parser.add_argument('--spread', type=float, default = 0.5)

    # COPE
    parser.add_argument('--beta', type=float, default = 1.5)
    parser.add_argument('--budget', action='store_true', default=False)

    return parser

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    
    return parser.parse_args(args)