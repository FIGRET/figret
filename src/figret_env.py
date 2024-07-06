import os
import json
import numpy as np
import torch

from networkx.readwrite import json_graph
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict

from .config import DATA_DIR
from .figret_simulator import FigretSimulator
from .utils import normalize_size

class FigretEnv():

    def __init__(self, props):
        """Initialize the FigretEnv with the properties.

        Args:
            props: arguments from the command line
        """
        self.topo_name = props.topo_name
        self.props = props
        self.init_topo()
        self.simulator = FigretSimulator(props, self.num_nodes)

    def init_topo(self):
        """Initialize the topology information with the given name."""
        self.G = self.read_graph_json(self.topo_name)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        self.adj = self.get_adj()
        self.pij = self.paths_from_file()
        self.edges_map, self.capacity = self.get_edges_map()
        self.paths_to_edges = self.get_paths_to_edges(self.pij)
        self.num_paths = self.paths_to_edges.shape[0]
        self.commodities_to_paths = self.get_commodities_to_paths()
        self.commodities_to_path_nums = self.get_commodities_to_path_nums()
        self.constant_pathlen = self.is_path_length_constant(self.commodities_to_path_nums)
        

    def set_mode(self, mode):
        """Set the mode of the simulator.

        Args:
            mode: train or test
        """
        self.simulator.set_mode(mode)

    def read_graph_json(self, topo_name):
        """Read the graph from the json file.

        Args:
            topo_name: name of the topology
        """
        with open(os.path.join(DATA_DIR, topo_name, topo_name + '.json'), 'r') as f:
            data = json.load(f)
        g = json_graph.node_link_graph(data)
        return g
    
    def paths_from_file(self):
        """Read the candidate paths from the file."""
        paths_file = "%s/%s/%s"%(DATA_DIR, self.topo_name, self.props.paths_file)
        pij = defaultdict(list)
        pid = 0
        with open(paths_file, 'r') as f:
            lines = sorted(f.readlines())
            lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
            for src in range(self.num_nodes):
                for dst in range(self.num_nodes):
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
                            pij[(i, j)].append(self.node_to_path(node_list))
                            pid += 1
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()
        return pij
    
    def node_to_path(self, node_list):
        """Convert the node list path to the edge list path."""
        return [(v1, v2) for v1, v2 in zip(node_list, node_list[1:])]
    
    def get_edges_map(self):
        """Get the map from the edge to the edge id."""
        eid = 0
        edges_map = dict()
        capacity = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj[i,j] == 1:
                    edges_map[(i,j)] = eid
                    capacity.append(normalize_size(self.G[i][j]['capacity']))
                    eid += 1
        return edges_map, capacity
    
    def get_adj(self):
        """Get the adjacency matrix of the graph."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d:
                    continue
                if d in self.G[s]:
                    adj[s,d] = 1
        return adj
    
    def get_paths_to_edges(self, paths):
        """Get the paths_to_edges matirx, [num_paths, num_edges]
           paths_to_edges[i, j] = 1 if edge j is in path i

        Args:
            paths: the candidate paths
        """
        paths_arr = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                for p in paths[(i, j)]:
                    p_ = [self.edges_map[e] for e in p]
                    p__ = np.zeros((int(self.num_edges),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
        return csr_matrix(np.stack(paths_arr))
    
    def get_commodities_to_paths(self):
        """Get the commodities_to_paths matrix, [num_commodities, num_paths]
           commodities_to_paths[i, j] = 1 if path j is a candidate path for commodity i
        """
        commodities_to_paths = lil_matrix((self.num_nodes * (self.num_nodes - 1), self.num_paths))
        commid = 0
        pathid = 0
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                for _ in self.pij[(src,dst)]:
                    commodities_to_paths[commid, pathid] = 1
                    pathid += 1
                commid += 1
        return csr_matrix(commodities_to_paths)

    def get_commodities_to_path_nums(self):
        """Get the number of candidate paths for each commodity."""
        path_num_per_commdities = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                path_num_per_commdities.append(len(self.pij[(src, dst)]))
        return path_num_per_commdities
    
    def is_path_length_constant(self, lst):
        """Check if all path len in the list are the same."""
        assert len(lst) > 0
        return lst.count(lst[0]) == len(lst)
