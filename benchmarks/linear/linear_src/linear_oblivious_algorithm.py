from .linear_routing import Routing
from .utils import Get_common_cases_tms

class oblivious_linear_algorithm(object):

    def __init__(self, props, topo, candidate_path, edge_to_path, tms) -> None:
        self.props = props
        self.routing = Routing(topo, candidate_path, edge_to_path)
        self.tms = tms
        
    
    def solve_traffic_engineering(self):
        pass

class oblivious(oblivious_linear_algorithm):
    """Oblivious TE algorithm."""
    def __init__(self, props, topo, candidate_path, edge_to_path, tms) -> None:
        super().__init__(props, topo, candidate_path, edge_to_path, tms)

    def solve_traffic_engineering(self):
        _, routing_weight = self.routing.dual_oblivious_traffic_engineering(self.tms)
        return _, routing_weight
    
class COPE(oblivious_linear_algorithm):
    """COPE TE algorithm."""
    def __init__(self, props, topo, candidate_path, edge_to_path, tms) -> None:
        super().__init__(props, topo, candidate_path, edge_to_path, tms)


    def solve_traffic_engineering(self):
        predict_tms = Get_common_cases_tms(self.tms)
        _, routing_weight = self.routing.dual_cope_traffic_engineering(self.tms, predict_tms, self.props.beta)
        return _, routing_weight