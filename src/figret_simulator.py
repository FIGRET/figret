import numpy as np
from tqdm import tqdm

from .utils import get_train_test_files, normalize_size

class FigretSimulator():

    def __init__(self, props, num_nodes):
        """Initialize the FigretSimulator.

        Args:
            props: arguments from the command line
            num_nodes: number of nodes in the topology
        """
        self.props = props
        self.num_nodes = num_nodes
        self.init_histories()
        self.tm_hist_std = self.get_tm_histories_std()

    def init_histories(self):
        """Initialize the histories traffic demand."""
        train_hist_file, test_hist_file = get_train_test_files(self.props)
        self.train_hist = Histories(train_hist_file, "train",self.num_nodes)
        self.test_hist = Histories(test_hist_file, "test", self.num_nodes)

    def get_tm_histories_std(self):
        """Get the standard deviation of the train traffic per s-d pairs."""
        tm_hist = np.array(self.train_hist.tms)
        tm_hist_std = np.std(tm_hist, axis=0)
        return tm_hist_std
    
    def set_mode(self, mode):
        """Set the mode of the simulator.

        Args:
            mode: train or test
        """
        self.cur_hist = self.train_hist if mode == "train" else self.test_hist
    
class Histories():

    def __init__(self, files, postfix, num_nodes):
        """Initialize the Histories with the files and postfix.

        Args:
            files: list of files
            postfix: train or test
            num_nodes: number of nodes in the topology
        """
        self.tms = []
        self.opts = []
        self.tm_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        np.fill_diagonal(self.tm_mask, 0)
        self.tm_mask = self.tm_mask.flatten()

        for fname in files:
            print('Population %s data from %s'%(postfix, fname))
            self.populate_tms(fname)
            self.read_opt(fname)
    
    def populate_tms(self, fname):
        """Populate the traffic matrices from the file.

        Args:
            fname: file name
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                tm = self.parse_tm_line(line)
                tm = normalize_size(tm)
                self.tms.append(tm)
    
    def parse_tm_line(self, line):
        """Parse the traffic matrix line.

        Args:
            line: line of traffic matrix
        """
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype = np.float64)
        return tm[self.tm_mask]
    
    def read_opt(self, fname):
        """Read the optimal paths from the file.

        Args:
            fname: file name
        """
        with open(fname.replace(".hist", ".opt"), 'r') as f:
            lines = f.readlines()
            self.opts += [np.float64(_) for _ in lines if _]
