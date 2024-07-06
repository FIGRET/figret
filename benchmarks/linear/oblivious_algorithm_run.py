import sys
import os
sys.path.append(os.path.join('..','..'))
from src.config import RESULT_DIR
from src.utils import print_to_txt

from linear_src.linear_oblivious_algorithm import oblivious, COPE
from linear_src.linear_env import LinearEnv
from linear_src.utils import Get_edge_to_path

from linear_helper import parse_args

def benchmark(props):
    env = LinearEnv(props)
    candidate_path = env.pij
    if props.budget:
        cut_candidate_path = {}
        for src in env.G.nodes:
            for dst in env.G.nodes:
                if src != dst:
                    cut_candidate_path[(src, dst)] = candidate_path[(src, dst)][:4]
        candidate_path = cut_candidate_path
    edge_to_path = Get_edge_to_path(env.G, candidate_path)
    if props.TE_solver == 'oblivious':
        algorithm = oblivious(props, env.G, candidate_path, edge_to_path, env.simulator.train_hist.tms)
    elif props.TE_solver == 'COPE':
        algorithm = COPE(props, env.G, candidate_path, edge_to_path, env.simulator.train_hist.tms)
    _, path_routing_weight = algorithm.solve_traffic_engineering()
    dm_list = env.simulator.test_hist.tms
    opt_list = env.simulator.test_hist.opts
    mlu_list = []
    result_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'result.txt')
    # Subtracting hist_len here to align with the 
    # results from window_algorithm for easier comparison.
    for index in range(len(dm_list)-props.hist_len):
        mlu = algorithm.routing.Get_MLU(path_routing_weight, dm_list[index+props.hist_len])
        mlu_list.append(mlu / opt_list[index+props.hist_len])
    
    print_to_txt(mlu_list, result_save_path)

if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)