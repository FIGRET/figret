import numpy as np
from tqdm import tqdm
from .utils import linear_get_dir, linear_get_hists_from_folder

class LinearSimulator(object):
    def __init__(self, props):
        self.props = props
        self.get_train_test_file(props)

    def get_train_test_file(self, props):
        data_train_folder = linear_get_dir(props, is_test = False)
        data_test_folder = linear_get_dir(props, is_test = True)

        train_hist_files = linear_get_hists_from_folder(data_train_folder)
        test_hist_files = linear_get_hists_from_folder(data_test_folder)

        self.train_hist = Histories(train_hist_files, 'train')
        self.test_hist = Histories(test_hist_files, 'test')

    def set_mode(self, mode):
        hist_str = 'self.' + mode + '_hist'
        self.cur_hist = eval(hist_str)

class Histories(object):

    def __init__(self,tm_files = None, htype = None):
        self.tms = []
        self.opts = []
        self.htype = htype

        for fname in tm_files:
            print('[+] Populating Tms for file: {}'.format(fname))
            self.populate_tms(fname)
            self.read_opts(fname)

    def read_opts(self, fname):
        try:
            with open(fname.replace('hist', 'opt')) as f:
                lines = f.readlines()
                self.opts += [np.float64(_) for _ in lines]
        except:
            return None
    
    def populate_tms(self, fname):
        with open(fname) as f:
            for line in tqdm(f.readlines()):
                try:
                    tm = self.parse_tm_line(line)
                except:
                    import pdb;
                    pdb.set_trace()
                
                self.tms.append(tm)

    def parse_tm_line(self, line):
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype = np.float64)
        return tm.flatten()