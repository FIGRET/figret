import os
from tqdm import tqdm
import sys
sys.path.append(os.path.join('..','..'))
from src.utils import print_to_txt
from src.config import RESULT_DIR

from linear_src.linear_env import LinearEnv
from linear_src.linear_window_algorithm import Jupiter
from linear_src.utils import Get_edge_to_path

from linear_helper import parse_args

def benchmark(props):
    env = LinearEnv(props)
    candidate_path = env.pij
    edge_to_path = Get_edge_to_path(env.G, candidate_path)
    if props.TE_solver == 'Jupiter':
        algorithm = Jupiter(props, env.G, candidate_path, edge_to_path)

    dm_list = env.simulator.test_hist.tms
    opt_list = env.simulator.test_hist.opts
    result_list = []
    result_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'result.txt')
    for index in tqdm(range(len(dm_list)-props.hist_len)):
        demands = dm_list[index:index+props.hist_len]
        _, path_routing_weight = algorithm.solve_traffic_engineering(demands)
        mlu = algorithm.routing.Get_MLU(path_routing_weight, dm_list[index+props.hist_len])
        result_list.append(mlu / opt_list[index+props.hist_len])
    print_to_txt(result_list, result_save_path)

if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)