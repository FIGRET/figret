## Getting Started (10 human-minutes + 10 compute-minutes)
1. Install the python packages which are listed in requirements.txt.
```sh
pip install -r requirements.txt
```
2. Install [Gurobi Solver](https://www.gurobi.com). You can request a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
3. Install `torch`, `torch-scatter` and `torch-sparse`. When installing `torch`, `torch-scatter`, and `torch-sparse`, it is essential to choose versions that are compatible with your execution environment (CPU or GPU with a specific version of CUDA). You can follow the [official instructions](https://pytorch.org/get-started/previous-versions/) to download them.

## Evaluating Figret
To evaluate figret:
```
$ python3 figret.py --topo_name Facebook_tor_a --epochs 3 --batch_size 32 --alpha 0.07
Population train data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/train/1.hist
100%|█████████████████████████| 2023/2023 [00:11<00:00, 182.91it/s]
Population train data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/train/2.hist
100%|█████████████████████████| 2023/2023 [00:11<00:00, 181.78it/s]
Population train data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/train/3.hist
100%|█████████████████████████| 2023/2023 [00:11<00:00, 181.44it/s]
Population train data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/train/4.hist
100%|█████████████████████████| 449/449 [00:02<00:00, 179.27it/s]
Population test data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/test/5.hist
100%|█████████████████████████| 2023/2023 [00:11<00:00, 180.72it/s]
Population test data from /home/gdp/lxm/FIGRET/src/../Data/Facebook_tor_a/test/6.hist
100%|█████████████████████████| 148/148 [00:00<00:00, 191.62it/s]
Epoch 1/3: 100%|█████████████████████████| 204/204 [00:38<00:00,  5.31it/s, loss_val=1.81]
Epoch 2/3: 100%|█████████████████████████| 204/204 [00:36<00:00,  5.66it/s, loss_val=1.66]
Epoch 3/3: 100%|█████████████████████████| 204/204 [00:36<00:00,  5.63it/s, loss_val=1.59]
```

## Evaluating benchmarks
Figret is compared with the following benchmarks:
- Jupiter (`benchmarks/linear/window_algorithm_run.py`) This scheme constructs an anticipated matrix composed of the peak values for each sourcedestination pair within a time window. Then it optimizes the TE objective under the constraint that path sensitivity remains below a predetermined threshold. (Requires Gurobi)
- Oblivious (`benchmarks/linear/oblivius_algorithm_run.py`) This scheme focuses on optimizing the worst-case performance across all possible traffic demands. (Requires Gurobi)
- COPE (`benchmarks/linear/oblivius_algorithm_run.py`) This scheme enhances demand-oblivious TE by also optimizing over a set of predicted traffic demands. It optimizes MLU across a set of DMs predicted based on previously observed DMs while retaining a worst-case performance guarantee. (Requires Gurobi)
- Pred TE (`benchmarks/linear/predict_algorithm_run.py`) This method involves predicting the next incoming traffic demand and configuring accordingly, without considering the mispredictions that may arise from the traffic uncertainty. (Requires Gurobi)

To evaluate these benchmarks, navigate to the benchmarks/linear directory and execute the following commands:
```
cd benchmarks/linear
python3 window_algorithm_run.py --topo_name Facebook_tor_a --TE_solver Jupiter
python3 oblivious_algorithm_run.py --topo_name Facebook_tor_a --TE_solver oblivious
python3 oblivious_algorithm_run.py --topo_name Facebook_tor_a --TE_solver COPE
python3 predict_algorithm_run.py --topo_name Facebook_tor_a
```
Additionally, there are two open-source benchmarks available, which are:
- **DOTE**: From the NSDI '23 paper titled "DOTE: Rethinking (Predictive) WAN Traffic Engineering". The code repository is available at [DOTE GitHub repository](https://github.com/PredWanTE/DOTE).

- **Teal**: From the Sigcomm '23 paper titled "Teal: Traffic Engineering Accelerated by Learning". The code repository is available at [Teal GitHub repository](https://github.com/harvard-cns/teal).

## Evaluating other topologies
If you have your own dataset and wish to test it, please organize your data according to the format outlined below and place it in the Data folder.

- topo_name.json: This file should include information about the network topology. It needs to specify whether the graph is directed, as well as details about the nodes and edges.

- tunnels.txt: This file should list the candidate paths for each source-destination pair. It serves as a key input for path selection in the network simulations.

- train/test folder: These folders should contain the demand matrices information: Each Demand Matrix (DM) should be flattened to a line and stored in a file with a .hist extension. Additionally, each file should include the optimal maximum-link-utilization (MLU) for each DM, which should be stored in a file with a .opt extension.

## Extending Figret
To add another TE implementation to this repo,
- If the implementation is based on machine learning, add test code to `figret.py` and source code to `src/`
- If the implementation is based on linear programming, add test code to `benchmarks/linear/` and source code to `benchmarks/linear/linear_src`