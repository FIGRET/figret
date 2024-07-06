from torch import nn

class FigretNetWork(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num):
        """Initialize the FigretNetWork with the network structure.

        Args:
            input_dim: dimension of input data, history len * flattened traffic matrix
            output_dim: dimension of output data, len of candidate paths all s-d pairs
            layer_num: number of hidden layers
        """
        super(FigretNetWork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, 128))
        self.layers.append(nn.ReLU())
        for _ in range(layer_num):
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, output_dim))
        self.layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        """Forward the input data through the network.

        Args:
            x: input data, history len * flattened traffic matrix
        """
        x = self.flatten(x)
        logits = self.net(x)
        return logits
    