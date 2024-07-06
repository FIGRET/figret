import os
import pickle
from tqdm import tqdm
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
sys.path.append(os.path.join('..', '..'))
from src.config import RESULT_DIR
from src.utils import print_to_txt

from linear_src.linear_env import LinearEnv
from linear_src.utils import Get_edge_to_path
from linear_src.linear_routing import Routing

from linear_helper import parse_args

def benchmark(props):
    env = LinearEnv(props)
    edge_to_path = Get_edge_to_path(env.G, env.pij)
    window_size = props.hist_len
    node_num = env.num_nodes

    # Load or train predict model
    predict_model_cache_path = os.path.join('predict_cache', f'{props.topo_name}.pkl')
    if os.path.exists(predict_model_cache_path):
        with open(predict_model_cache_path, 'rb') as f:
            predict_models = pickle.load(f)
    else:
        predict_models = {}
        train_tms = env.simulator.train_hist.tms

        for i in tqdm(range(node_num), desc='train predict model'):
            for j in range(node_num):
                if i != j:
                    X, Y = [], []
                    for t in range(len(train_tms) - window_size):
                        X.append([train_tms[t+k][i * node_num + j] for k in range(window_size)])
                        Y.append(train_tms[t+window_size][i * node_num + j])
                    X, Y = np.array(X),np.array(Y)
                    predict_model = LinearRegression(n_jobs=-1).fit(X, Y)
                    predict_models[(i, j)] = predict_model
        if not os.path.exists(os.path.dirname(predict_model_cache_path)):
            os.makedirs(os.path.dirname(predict_model_cache_path))
        with open(predict_model_cache_path, 'wb') as f:
            pickle.dump(predict_models, f)

    # Test
    algorithm = Routing(env.G, env.pij, edge_to_path)
    tm_list = env.simulator.test_hist.tms
    opt_list = env.simulator.test_hist.opts
    mlu_list = []
    for index in tqdm(range(len(tm_list) - window_size)):
        pre_tm = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(node_num):
                if i != j:
                    pre_tm[i][j] = predict_models[(i,j)].predict(np.array([[tm_list[index + k][i * node_num + j] for k in range(window_size)]])).flatten()[0]
        _, path_routing_weight = algorithm.MLU_traffic_engineering([pre_tm])
        mlu = algorithm.Get_MLU(path_routing_weight, tm_list[index + window_size])
        mlu_list.append(mlu / opt_list[index + window_size])
    
    result_save_path = os.path.join(RESULT_DIR, props.topo_name, 'predict', 'result.txt')
    print_to_txt(mlu_list, result_save_path)

if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)
