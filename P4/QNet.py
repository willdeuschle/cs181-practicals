import numpy as np
import torch
from torch import nn, cat
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor, FloatTensor
from torch.autograd import Variable
from graphviz import Digraph
from copy import deepcopy
from torch import optim

class QNet(nn.Module):
    def __init__(self, structure, initializations=None):
        super().__init__()
        self.stored_layers = OrderedDict()
        for idx, input_output_tuple in enumerate(structure):
            self.stored_layers[str(idx)] = nn.Linear(input_output_tuple[0], input_output_tuple[1])
        self.layers = nn.Sequential(self.stored_layers)
        # for optimizing
        self.optimizer = optim.Adam(self.parameters(), lr=.0001)
        # create a loss function
        self.criterion = nn.SmoothL1Loss()
    def forward(self, x):
        return self.layers(x)
    def do_backprop(self, net_out, target):
        # self.optimizer.zero_grad()
        loss = self.criterion(net_out, target)
        print("what is loss", loss)
        loss.backward()
        self.optimizer.step()
        return
    def visualize_net(self, small=False):
        dot = Digraph(comment='Neural Net Visualization')
        layers = list()
        for idx, layer in net.stored_layers.items():
            if idx == '0':
                if small:
                    layers.append(int(np.log(layer.in_features)))
                else:
                    layers.append(layer.in_features)
            if small:
                layers.append(int(np.log(layer.out_features)))
            else:
                layers.append(layer.out_features)
        # create the nodes
        for layer_idx, num_nodes in enumerate(layers):
            for node_idx in range(num_nodes):
                dot.node(f"L{layer_idx}N{node_idx}", f"L{layer_idx}N{node_idx}")
        # now create the edges
        for layer_idx, num_nodes in enumerate(layers):
            if layer_idx == 0:
                continue
            prev_layer_idx = layer_idx - 1
            prev_layer_num_nodes = layers[prev_layer_idx]
            for prev_node_idx in range(prev_layer_num_nodes):
                for current_node_idx in range(num_nodes):
                    dot.edge(f"L{prev_layer_idx}N{prev_node_idx}", f"L{layer_idx}N{current_node_idx}", constraint='true')
        return dot

def init_network(structure, initializations=None):
    # assume structure is an array from input layer to final output layer
    # for layer_spec in structure:
    net = QNet(structure)
    return net
