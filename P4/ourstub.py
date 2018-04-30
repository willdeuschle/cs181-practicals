EPSILON_START = 0.95
EPSILON_END = 0.05
EPSILON_DECAY = 3000


# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey
from QNet import QNet, init_network
import torch
from torch.autograd import Variable


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.is_first = True
        self.gravity = 0
        self.actions_taken = 0
        self.q_net = init_network([(7,10), (10,10), (10,7), (7,2)])

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.is_first = True
        self.gravity = 0

    def format_state(self, state):
        vector_state = list()
        vector_state.append(self.gravity)
        vector_state.append(state['tree']['dist'])
        vector_state.append(state['tree']['top'])
        vector_state.append(state['tree']['bot'])
        vector_state.append(state['monkey']['vel'])
        vector_state.append(state['monkey']['top'])
        vector_state.append(state['monkey']['bot'])
        vector_state = torch.Tensor([vector_state])
        vector_state = Variable(vector_state)
        return vector_state

    def compute_loss(self):
        return (self.last_reward + (GAMMA * self.predicted_reward) - self.previous_predicted_reward)**2

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        if self.is_first:
            self.is_first = False
            self.predicted_reward = self.q_net(self.format_state(state))
            self.last_state = state
            self.last_action = torch.tensor([npr.random() > 0.25], dtype=torch.long)
            return self.last_action
        # if self.is_first:
            # # calculate gravity, etc.
            # self.first_position = state['monkey']['top']
            # self.is_first = False
            # self.predicted_reward = self.q_net(self.format_state(state))
            # # return 1 to get sense of gravity
            # return 1
        # if self.gravity is None:
            # # we're in the second step, can now compute gravity
            # self.gravity = state['monkey']['top'] - self.first_position

        # if we jumped last time, figure out what gravity is
        if self.last_action == 1:
            self.gravity = state['monkey']['top'] - self.last_state['monkey']['top']

        # get epsilon decision
        rand_val = npr.random()
        threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.actions_taken / EPSILON_DECAY)
        print("what is threshold", threshold)
        self.actions_taken += 1

        # epsilon greedy decision here, get next action
        if rand_val > threshold:
            with torch.no_grad():
                # print("HERERERERERERERERER")
                new_action = torch.tensor([self.q_net(self.format_state(state))[0].max(0)[1]])
        else:
            new_action = torch.tensor([npr.random() > 0.85], dtype=torch.long)

        # store our previous prediction
        self.previous_predicted_reward = self.predicted_reward

        # LEARNING

        # You might do some learning here based on the current state and the last state.
        # get next action
        # no_action_reward = self.q_net(self.format_state(state))
        # yes_action_reward = self.q_net(self.format_state(state))
        # get next action
        # new_action = None
        # if yes_action_reward >= no_action_reward:
            # new_action = 1
            # self.predicted_reward = yes_action_reward
        # else:
            # new_action = 0
            # self.predicted_reward = no_action_reward

        # push this pair to memory
        self.q_net.memory.push(self.format_state(self.last_state), self.last_action, self.format_state(state), torch.tensor([self.last_reward]))
        # LEARN DQN
        self.q_net.train()
        # expected_reward = Variable((GAMMA * self.predicted_reward) + self.last_reward, requires_grad=False)
        # # backprop to update network
        # self.q_net.do_backprop(self.previous_predicted_reward, expected_reward)

        # END LEARNING

        self.last_action = new_action
        self.last_state  = state

        # print("ACTION", self.last_action)
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        # print("REWARD", reward)

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	# run_games(agent, hist, 20, 10)
	run_games(agent, hist, 1000, 20)

	# Save history.
	np.save('hist',np.array(hist))


