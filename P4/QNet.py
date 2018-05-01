import numpy as np
import random
import torch
from torch import nn, cat
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor, FloatTensor
from torch.autograd import Variable
from graphviz import Digraph
from copy import deepcopy
from torch import optim
from collections import namedtuple

BATCH_SIZE = 64
GAMMA = 0.999

# borrowed from pytorch
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# this is taken from a pytorch example, we didn't end up using this stuff
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    def __init__(self, structure, initializations=None):
        super().__init__()
        self.stored_layers = OrderedDict()
        for idx, input_output_tuple in enumerate(structure):
            self.stored_layers[str(idx)] = nn.Linear(input_output_tuple[0], input_output_tuple[1])
            # TODO
            # if idx == len(structure) - 1:
                # self.stored_layers[str(idx)] = nn.Linear(input_output_tuple[0], input_output_tuple[1])
            # else:
                # self.stored_layers[str(idx)] = F.relu(input_output_tuple[0], input_output_tuple[1])

        self.layers = nn.Sequential(self.stored_layers)
        # for optimizing
        self.optimizer = optim.Adam(self.parameters(), lr=.00005)
        # self.optimizer = optim.RMSprop(self.parameters())
        # store memory
        self.memory = ReplayMemory(10000)
        # create a loss function
        self.criterion = nn.SmoothL1Loss()
    def forward(self, x):
        return self.layers(x)
    def train(self):
        # this is taken from a pytorch example, we didn't end up using this stuff
        if len(self.memory) < BATCH_SIZE:
              return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
					      batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
						    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self(state_batch).gather(1, action_batch.view(-1,1))
        # print("huh", state_action_values)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print('what is loss', loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.parameters():
            # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return
    def do_backprop(self, net_out, target):
        # self.optimizer.zero_grad()
        loss = self.criterion(net_out, target)
        # print("what is loss", loss)
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
