import os
import glob

from .config import DATA_DIR

class SizeConsts:
    """Constants for sizes. 
       Used for normalizing the inputs of a neural network.
    """
    ONE_BIT = 1
    ONE_BYTE = 8 * ONE_BIT
    ONE_KB = 1024 * ONE_BYTE
    ONE_MB = 1024 * ONE_KB
    ONE_GB = 1024 * ONE_MB
    
    ONE_Kb = 1000 * ONE_BIT
    ONE_Mb = 1000 * ONE_Kb
    ONE_Gb = 1000 * ONE_Mb
    
    GB_TO_MB_SCALE = ONE_Gb / ONE_Mb
    
    BPS_TO_GBPS = lambda x: x / SizeConsts.ONE_Gb
    GBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Gb
    
    GBPS_TO_MBPS = lambda x: x * SizeConsts.GB_TO_MB_SCALE
    
    BPS_TO_MBPS = lambda x: x / SizeConsts.ONE_Mb
    MBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Mb

    BPS_TO_KBPS = lambda x: x / SizeConsts.ONE_Kb
    KBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Kb

def normalize_size(x):
    """Normalize the input size.
       To prevent excessively large input data from impacting training, 
       we use BPS_TO_GBPS to normalize the capacities and flows of edges. 
       If your data is very small, you can modify this, 
       such as changing it to BPS_TO_MBPS.
    """
    return SizeConsts.BPS_TO_GBPS(x)

def get_dir(props, is_test):
    """Get the train or test directory for the given topology."""
    postfix = "test" if is_test else "train"
    return os.path.join(DATA_DIR, props.topo_name, postfix)

def get_hists_from_folder(folder):
    """Get the list of histogram files from the given folder."""
    hists = sorted(glob.glob(folder + "/*.hist"))
    return hists

def get_train_test_files(props):
    """Get the train and test files for the given properties."""
    train_dir = get_dir(props, False)
    test_dir = get_dir(props, True)
    train_files = get_hists_from_folder(train_dir)
    test_files = get_hists_from_folder(test_dir)
    return sorted(train_files), sorted(test_files)

def print_to_txt(result, path):
    """Print the result to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        dists = [float(v) for v in result]
        for item in dists:
            f.write('%s\n' %(item))
    f.close()


