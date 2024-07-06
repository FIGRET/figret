from .linear_routing import Routing
from .utils import Get_peak_demand

class window_linear_algorithm(object):

    def __init__(self, props, topo, candidate_path, edge_to_path):
        self.props = props
        self.routing = Routing(topo, candidate_path, edge_to_path)

    def solve_traffic_engineering(self):
        pass

class Jupiter(window_linear_algorithm):
    """Google's TE algorithm in the Jupiter DCN."""

    def __init__(self, props, topo, candidate_path, edge_to_path):
        super(Jupiter, self).__init__(props, topo, candidate_path, edge_to_path)

    def solve_traffic_engineering(self, demands):
        peak_demand = Get_peak_demand(demands)
        mlu, path_routing_weight = self.routing.Spread_traffic_engineering(peak_demand, self.props.spread)
        return mlu, path_routing_weight 