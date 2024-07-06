import numpy as np
import os
import glob
from collections import defaultdict
import sys  
from sklearn.cluster import KMeans
sys.path.append(os.path.join('..','..'))

from src.config import DATA_DIR

def Get_peak_demand(dm_list):
    """Get the peak demand from the demand matrix."""
    dm_matrixs = np.array(dm_list)
    predict_dm = np.max(dm_matrixs, axis=0)
    return predict_dm

def Get_edge_to_path(topology,candidate_path):
    """ Get the mapping from edge to path."""
    edge_to_path = {}
    for edge in topology.edges:
        edge_to_path[(int(edge[0]),int(edge[1]))] = []
    for src in topology.nodes:
        for dst in topology.nodes:
            if src != dst:
                for index, path in enumerate(candidate_path[(src, dst)]):
                    for i in range(len(path) - 1):
                        edge_to_path[(int(path[i]), int(path[i + 1]))].append((int(src), int(dst), index))
    return edge_to_path

def linear_get_dir(props, is_test):
    """Get the train or test directory for the given topology."""
    postfix = "test" if is_test else "train"
    return os.path.join(DATA_DIR,props.topo_name, postfix)

def linear_get_hists_from_folder(folder):
    """Get the list of histogram files from the given folder."""
    hists = sorted(glob.glob(folder + "/*.hist"))
    return hists

def paths_from_file(paths_file, num_nodes):
    """Get the paths from the file."""
    pij = defaultdict(list)
    pid = 0
    with open(paths_file, 'r') as f:
        lines = sorted(f.readlines())
        lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                try:
                    if "%d %d" % (src, dst) in lines_dict:
                        line = lines_dict["%d %d" % (src, dst)].strip()
                    else:
                        line = [l for l in lines if l.startswith("%d %d:" % (src, dst))]
                        if line == []:
                            continue
                        line = line[0]
                        line = line.strip()
                    if not line: continue
                    i,j = list(map(int, line.split(":")[0].split(" ")))
                    paths = line.split(":")[1].split(",")
                    for p_ in paths:
                        node_list = list(map( int, p_.split("-")))
                        pij[(i, j)].append(node_list)
                        pid += 1
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
    return pij

def Get_common_cases_tms(hist_tms):
    """Get the common cases traffic demands from the history traffic demands.

    Args:
        hist_tms: the history traffic demands.
    """
    # Computing the convex hull can be challenging when the length of hist_tms is particularly large, 
    # or when the dimensionality of each tm is very high. 
    # Therefore, we use all hist_tms as common_cases_tms.

    # hull = ConvexHull(hist_tms)
    # hull_vertices = hull.vertices
    # common_case_tms = [hist_tms[i] for i in hull_vertices]
    common_case_tms = hist_tms
    return common_case_tms
