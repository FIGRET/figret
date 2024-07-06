import numpy as np
import gurobipy as grb


class Routing:
    def __init__(self, topology, candidate_path, edge_to_path):
        """Initialize the routing with the topology, candidate paths and edge.
        
        Args:
            topology: the network topology
            candidate_path: the candidate paths for each s-d pair
            edge_to_path: the mapping from edge to path
        """
        self.topology = topology
        self.candidate_path = candidate_path
        self.edge_to_path = edge_to_path
        self.path_capacity = self.Get_path_capacity()
        
    def MLU_traffic_engineering(self, demands):
        """Compute the traffic engineering solutions for multiple demands to minimize the worst-case MLU.

        Args:
            demands: the traffic demands, shape: (demand_number, number_of_nodes * number_of_nodes)
        """
        for i in range(len(demands)):
            demands[i] = demands[i].reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0
        mlu = m.addVar(lb = 0, vtype = grb.GRB.CONTINUOUS, name = 'mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i
                    for k in range(len(self.candidate_path[(i, j)]))
                    ]
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # the sum of the routing weights of the candidate paths for each s-d pair should be 1
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        
        for demand in demands:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= mlu * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )
        
        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')

    def Spread_traffic_engineering(self, demand, Spread):
        """Compute the traffic engineering solutions for a single demand to minimize the worst-case MLU with Google's hedging mechanism.

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
        """
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0

        mlu = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
            for k in range(len(self.candidate_path[(i, j)]))
            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')
        
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']
            for edge in self.topology.edges
        )
        m.addConstrs(
            path_weight[f'w_{i}_{j}_{k}'] * (Spread * grb.quicksum(
                    self.path_capacity[(i, j, k)] for k in range(len(self.candidate_path[(i, j)]))
                )) <=  
                self.path_capacity[(i, j, k)] 
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
            for k in range(len(self.candidate_path[(i, j)]))
        )
        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')

    def dual_oblivious_traffic_engineering(self,demands):
        """Compute the traffic engineering solutions in the oblivious model"""
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0
        ratio = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='obli_ratio')
        
        edge_dict = {}
        edge_list = []
        
        for edge in self.topology.edges:
            edge_list.append(edge)

        edge_src_dst_to_k = {} 
        for l in range(self.topology.number_of_edges()):
            for (src,dst,k) in self.edge_to_path[edge_list[l]]:
                if (l,src,dst) not in edge_src_dst_to_k:
                    edge_src_dst_to_k[(l,src,dst)] = [k]
                else:
                    edge_src_dst_to_k[(l,src,dst)].append(k)

        for i in range(self.topology.number_of_edges()):
            edge_dict[edge_list[i]] = i
        # Network Configuration
        name_path_weight = [f'w_{i}_{j}_{k}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i
                    for k in range(len(self.candidate_path[(i, j)]))
                    ]
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # split ratio on ratio, more detial in Making Intra-Domain Routing Robust to Changing
        name_f_dict = [f'f_{l}_{i}_{j}'
                       for l in range(self.topology.number_of_edges())
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i
                    ]
        f_dict = m.addVars(name_f_dict, lb=0, vtype=grb.GRB.CONTINUOUS, name='f_dict')

        # pi in Making Intra-Domain Routing Robust to Changing
        name_pi = [f'pi_{i}_{j}'
                    for i in range(self.topology.number_of_edges())
                    for j in range(self.topology.number_of_edges())]
        pi = m.addVars(name_pi, lb=0, vtype=grb.GRB.CONTINUOUS, name='pi')

        # p in Making Intra-Domain Routing Robust to Changing
        name_p = [f'p_{i}_{j}_{l}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    for l in range(self.topology.number_of_edges())]
        p = m.addVars(name_p, lb=0, vtype=grb.GRB.CONTINUOUS, name='p')

        # Network configuration constraints
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        
        # put the path split on edge        
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if (l,i,j) in edge_src_dst_to_k.keys():
                        m.addConstr(
                            f_dict[f'f_{l}_{i}_{j}'] == grb.quicksum(path_weight[f'w_{i}_{j}_{k}'] for k in edge_src_dst_to_k[(l,i,j)])
                        )

        # first constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in range(self.topology.number_of_edges())) <= ratio
            )
                
        # second constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i!=j:
                        if f'f_{l}_{i}_{j}' in f_dict:
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]]['capacity']
                            )
        
        # third constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )
        
        # fifth constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        # fourth and sixth constraint is in addVars lb.
        
        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution')

    def dual_cope_traffic_engineering(self, demands, predict_dms, beta):
        for idx, demand in enumerate(predict_dms):
            predict_dms[idx] = demand.reshape(self.topology.number_of_nodes(), self.topology.number_of_nodes())
        oblivious_ratio, _ = self.dual_oblivious_traffic_engineering(demands)
        plenalty_ratio = beta * oblivious_ratio
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0
        ratio = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='obli_ratio')
        
        edge_dict = {}
        edge_list = []
        
        for edge in self.topology.edges:
            edge_list.append(edge)

        edge_src_dst_to_k = {} 
        for l in range(self.topology.number_of_edges()):
            for (src,dst,k) in self.edge_to_path[edge_list[l]]:
                if (l,src,dst) not in edge_src_dst_to_k:
                    edge_src_dst_to_k[(l,src,dst)] = [k]
                else:
                    edge_src_dst_to_k[(l,src,dst)].append(k)

        for i in range(self.topology.number_of_edges()):
            edge_dict[edge_list[i]] = i
        # Network Configuration
        name_path_weight = [f'w_{i}_{j}_{k}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i
                    for k in range(len(self.candidate_path[(i, j)]))
                    ]
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # split ratio on ratio, more detial in Making Intra-Domain Routing Robust to Changing
        name_f_dict = [f'f_{l}_{i}_{j}'
                       for l in range(self.topology.number_of_edges())
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    if j != i
                    ]
        f_dict = m.addVars(name_f_dict, lb=0, vtype=grb.GRB.CONTINUOUS, name='f_dict')

        # pi in Making Intra-Domain Routing Robust to Changing
        name_pi = [f'pi_{i}_{j}'
                    for i in range(self.topology.number_of_edges())
                    for j in range(self.topology.number_of_edges())]
        pi = m.addVars(name_pi, lb=0, vtype=grb.GRB.CONTINUOUS, name='pi')

        # p in Making Intra-Domain Routing Robust to Changing
        name_p = [f'p_{i}_{j}_{l}'
                    for i in range(self.topology.number_of_nodes())
                    for j in range(self.topology.number_of_nodes())
                    for l in range(self.topology.number_of_edges())]
        p = m.addVars(name_p, lb=0, vtype=grb.GRB.CONTINUOUS, name='p')

        # Network configuration constraints
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        
        # put the path split on edge
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if (l,i,j) in edge_src_dst_to_k.keys():
                        m.addConstr(
                            f_dict[f'f_{l}_{i}_{j}'] == grb.quicksum(path_weight[f'w_{i}_{j}_{k}'] for k in edge_src_dst_to_k[(l,i,j)])
                        )

        # first constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in range(self.topology.number_of_edges())) <= plenalty_ratio
            )
                
        # second constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i!=j:
                        if f'f_{l}_{i}_{j}' in f_dict:
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]]['capacity']
                            )
        
        # third constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )
        
        # fifth constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        for demand in predict_dms:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= ratio * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )
        
        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution')

    def Get_path_capacity(self):
        """Get the capacity of the paths."""
        path_capacity = {}
        for src in range(self.topology.number_of_nodes()):
            for dst in range(self.topology.number_of_nodes()):
                if dst != src:
                    for index, path in enumerate(self.candidate_path[(src, dst)]):
                        path_capacity[(src, dst, index)] = min([self.topology.edges[(path[i], path[i + 1])]['capacity'] \
                                             for i in range(len(path) - 1)])
        return path_capacity
    
    def Get_MLU(self, path_weights, demand):
        """Get the MLU of the traffic engineering solution.

        Args:
            path_weights: the routing weights of the paths
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
        """
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        mlu = 0
        for edge in self.topology.edges:
            edge_lu = 0
            for (src, dst, k) in self.edge_to_path[edge]:
                edge_lu += path_weights[f'w_{src}_{dst}_{k}'] * demand[src][dst] / float(self.topology.edges[edge]['capacity'])
            mlu = max(mlu, edge_lu)
        return mlu

    def Spread_traffic_engineering_link_faliure(self, demand, Spread, link_faliure_list):
        """Compute the traffic engineering solutions for fault-aware traffic engineering

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
            link_faliure_list: the list of failed links
        """
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0
        name_path_weight = [f'w_{i}_{j}_{k}'
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
            for k in range(len(self.candidate_path[(i, j)]))
            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub = 1, vtype=grb.GRB.CONTINUOUS, name='path_weight')
        mlu = m.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']
            for edge in self.topology.edges
        )

        faliure_od_dict = {}
        for edge_id in link_faliure_list:
            # print(list(self.topology.edges)[edge_id])
            for (src, dst, k) in self.edge_to_path[list(self.topology.edges)[edge_id]]:
                # print((src,dst,k))
                faliure_od_dict[(src, dst)] = True
                m.addConstr(
                    path_weight[f'w_{src}_{dst}_{k}']  == 0 
                )

        m.addConstrs(
            path_weight[f'w_{i}_{j}_{k}'] * (Spread * grb.quicksum(
                    self.path_capacity[(i, j, k)] for k in range(len(self.candidate_path[(i, j)]))
                )) <=  
                self.path_capacity[(i, j, k)] 
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i and not faliure_od_dict.get((i, j), False)
            for k in range(len(self.candidate_path[(i, j)]))
        )

        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')
     