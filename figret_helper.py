import argparse

def add_default_args(parser):

    parser.add_argument('--topo_name', type=str, default = 'Facebook_tor_a')
    parser.add_argument('--paths_file', type=str, default = 'tunnels.txt')
    
    parser.add_argument('--mode', type=str, default = 'train')
    parser.add_argument('--batch_size', type=int, default = 1)
    parser.add_argument('--epochs', type=int, default = 1)
    parser.add_argument('--num_layer', type=int, default = 3)

    parser.add_argument('--hist_len', type=int, default = 12)
    parser.add_argument('--alpha', type=float, default = 0.03)

    parser.add_argument('--opt_name', type=str, default = '')

    return parser

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    
    return parser.parse_args(args)