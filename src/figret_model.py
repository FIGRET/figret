import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import os

from .config import RESULT_DIR, MODEL_DIR
from .utils import print_to_txt

class Figret():
    """Figret class for training and testing the model."""
    def __init__(self, props, env, device):
        """Initialize the Figret with the properties, environment, model and device.

        Args:
            props: arguments from the command line
            env: environment for the Figret
            model: model for the Figret
            device: GPU or CPU
        """
        self.props = props
        self.env = env
        self.device = device
        ctp_coo = env.commodities_to_paths.tocoo()
        self.commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), \
                                               torch.DoubleTensor(ctp_coo.data), 
                                               torch.Size(ctp_coo.shape)).to(device) # shape: (num_commodities, num_paths)
        pte_coo = env.paths_to_edges.tocoo()
        self.paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), \
                                               torch.DoubleTensor(pte_coo.data), \
                                               torch.Size(pte_coo.shape)).to(device) # shape: (num_paths, num_edges)
        self.tm_hist_std = torch.tensor(env.simulator.get_tm_histories_std()).to(device) # shape: (num_nodes * (num_nodes - 1),)
        self.edges_capacity = torch.tensor(env.capacity).unsqueeze(1).to(device) # shape: (num_edges, 1)

    def loss(self, y_pred_batch, y_true_batch):
        """Compute the loss of the model.

        Args:
            y_pred: the split ratios for the candidate paths
            y_true: the true traffic demand and the optimal mlu
        """
        num_nodes = self.env.num_nodes
        losses = []
        loss_vals = []
        batch_size = y_pred_batch.shape[0]
        for i in range(batch_size):
            y_pred = y_pred_batch[[i]]
            y_true = y_true_batch[[i]]
            opt = y_true[0][num_nodes * (num_nodes -1)].item()
            y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1)) #shape: (1, num_commodities)

            y_pred = y_pred + 1e-16
            paths_weight = torch.transpose(y_pred, 0, 1) #shape: (num_paths, 1)
            commodity_total_weight = self.commodities_to_paths.matmul(paths_weight) #shape: (num_commodities, 1)
            paths_over_total = self.commodities_to_paths.transpose(0, 1).matmul(1.0 / commodity_total_weight) #shape: (num_paths, 1)
            split_ratios = paths_weight.mul(paths_over_total) #shape: (num_paths, 1)
            tmp_demand_on_paths = self.commodities_to_paths.transpose(0, 1).matmul(y_true.transpose(0, 1)) #shape: (num_paths, 1)
            demand_on_paths = tmp_demand_on_paths.mul(split_ratios) #shape: (num_paths, 1)
            flow_on_edges = self.paths_to_edges.transpose(0, 1).matmul(demand_on_paths) #shape: (num_edges, 1)
            congestion = flow_on_edges.divide(self.edges_capacity) #shape: (num_edges, 1)
            max_cong = torch.max(congestion.flatten(), dim = 0).values

            if self.env.constant_pathlen:
                max_sensitivity = torch.max(split_ratios.view(num_nodes * (num_nodes - 1), -1), dim = 1).values #shape: (num_commodities,)
            else:
                split_ratios = torch.split(split_ratios, self.env.commodities_to_path_nums)
                max_sensitivity = torch.tensor([torch.max(split_ratio) for split_ratio in split_ratios]).to(self.device) #shape: (num_commodities,)
            weight_max_sensitivity = max_sensitivity.mul(self.tm_hist_std) #(num_commodities,)
            sum_wm_sens = torch.sum(weight_max_sensitivity) / len(weight_max_sensitivity)

            # loss function, the first term is the congestion, the second term is the sensitivity.
            # The operation of dividing by item() is used to balance different objectives, 
            # ensuring they are on the same scale. Then, alpha is used to adjust their importance.
            loss = 1.0 - max_cong if max_cong.item() == 0.0 else \
                    max_cong / max_cong.item() + self.props.alpha * sum_wm_sens / sum_wm_sens.item()
            loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt
            losses.append(loss)
            loss_vals.append(loss_val)
        
        ret = sum(losses) / len(losses)
        ret_val = sum(loss_vals) / len(loss_vals)
        return ret, ret_val
        
    def train(self, train_dl, model, optimizer, device):
        """Train the model with the given data.
        
        Args:
            train_dl: the train data loader
            model: the model for training
            optimizer: the optimizer for the model
            device: GPU or CPU
        """
        model = model.to(device)
        model_save_path = os.path.join(MODEL_DIR, f'{self.props.topo_name}_{self.props.opt_name}.pt') \
                if self.props.opt_name else os.path.join(MODEL_DIR, f'{self.props.topo_name}.pt')
        model.train()
        for epoch in range(self.props.epochs):
            with tqdm(train_dl) as tepoch:
                for inputs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}/{self.props.epochs}")
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model(inputs)
                    loss, loss_val = self.loss(preds, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss_val=loss_val)
        torch.save(model, model_save_path)
    
    def test(self, test_dl, model, device):
        """Test the model with the given data.

        Args:
            test_dl: the test data loader
            model: the model for testing
            device: GPU or CPU
        """
        model = model.to(device)
        result_save_path = os.path.join(RESULT_DIR, self.props.topo_name, 'Figret', 'result.txt')
        model.eval()
        with torch.no_grad():
            with tqdm(test_dl) as tests:
                test_losses = []
                for inputs, targets in tests:
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model(inputs)
                    _, loss_val = self.loss(preds, targets)
                    test_losses.append(loss_val)
        
        print_to_txt(test_losses, result_save_path)

class FigretDataset(Dataset):
    """Dataset for the FigretNetWork."""
    def __init__(self, props, env, mode):
        env.set_mode(mode)
        tms = env.simulator.cur_hist.tms
        opts = env.simulator.cur_hist.opts
        assert len(tms) == len(opts)
        tms = [np.asarray([tm]) for tm in tms]
        np_tms = np.vstack(tms) # shape: (num_samples, num_nodes * (num_nodes - 1))
        X = []
        for histid in range(len(tms) - props.hist_len):
            X.append(np_tms[histid:histid + props.hist_len].flatten())
        self.X = np.asarray(X) # shape: (num_samples, hist_len * num_nodes * (num_nodes - 1))
        self.Y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(tms))]) 
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]