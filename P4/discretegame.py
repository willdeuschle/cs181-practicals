# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

BIN_SIZE_HEIGHT = 25
BIN_SIZE_WIDTH = 100
NUM_VEL_BINS = 10
NUM_GRAVITY_BINS = 10
GAMMA = 0.9
LEARNING_RATE = 0.05
EPSILON_START = 0.95
EPSILON_END = 0.05
EPSILON_DECAY = 3000
SCREEN_WIDTH  = 600
SCREEN_HEIGHT = 400


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.max_value = 0
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.actions_taken = 0
        self.is_first = True
        self.gravity = 0
        # actions x tree dist x tree top x tree bottom x monkey vel x monkey top x monkey bottom x gravity
        self.Q_values = np.zeros(
            (2, # actions
             int(SCREEN_WIDTH / BIN_SIZE_WIDTH), # tree dist
             int(SCREEN_HEIGHT / BIN_SIZE_HEIGHT), # monkey top to tree top
             int(SCREEN_HEIGHT / BIN_SIZE_HEIGHT),# monkey bottom to tree bottom
             NUM_VEL_BINS, # monkey vel
             NUM_GRAVITY_BINS, # gravity
            2, # velocity pos/neg
            2, # monkey top above tree top
            2, # monkey bottom above tree bottom
        ))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.is_first = True
        self.gravity = 0

    def get_q_value_indexes(self, state):
        # get this state in terms of Q
        tree_dist = int(state['tree']['dist'] / BIN_SIZE_WIDTH)
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']

        monkey_vel = state['monkey']['vel']
        vel_pos = int(monkey_vel >= 0)
        monkey_vel = abs(monkey_vel)
        monkey_vel_idx = int(monkey_vel / NUM_VEL_BINS)
        gravity = int(state['gravity'] / NUM_GRAVITY_BINS)

        monkey_top_to_tree_top = tree_top - monkey_top
        monkey_top_below = int(monkey_top_to_tree_top >= 0)
        monkey_top_to_tree_top = int(abs(monkey_top_to_tree_top) / BIN_SIZE_HEIGHT)

        monkey_bot_to_tree_bot = monkey_bot - tree_bot
        monkey_bot_above = int(monkey_bot_to_tree_bot >= 0)
        monkey_bot_to_tree_bot = int(abs(monkey_bot_to_tree_bot) / BIN_SIZE_HEIGHT)

        return (tree_dist, monkey_top_to_tree_top, monkey_bot_to_tree_bot, monkey_vel_idx, gravity, vel_pos, monkey_top_below, monkey_bot_above)
        # return (tree_dist, tree_top, tree_bot, monkey_top, monkey_bot, monkey_vel_idx, gravity)

    def get_q_values(self, state):
        # tree_dist, tree_top, tree_bot, monkey_top, monkey_bot, monkey_vel_idx, gravity = self.get_q_value_indexes(state)
        tree_dist, monkey_top_to_tree_top, monkey_bot_to_tree_bot, monkey_vel_idx, gravity, vel_pos, monkey_top_below, monkey_bot_above = self.get_q_value_indexes(state)

        # no_jump_q_val = self.Q_values[0][tree_dist][tree_top][tree_bot][monkey_vel_idx][monkey_top][monkey_bot][gravity]
        # jump_q_val = self.Q_values[1][tree_dist][tree_top][tree_bot][monkey_vel_idx][monkey_top][monkey_bot][gravity]
        no_jump_q_val = self.Q_values[0][tree_dist][monkey_top_to_tree_top][monkey_bot_to_tree_bot][monkey_vel_idx][gravity][vel_pos][monkey_top_below][monkey_bot_above]
        jump_q_val = self.Q_values[1][tree_dist][monkey_top_to_tree_top][monkey_bot_to_tree_bot][monkey_vel_idx][gravity][vel_pos][monkey_top_below][monkey_bot_above]
        return (no_jump_q_val, jump_q_val)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        if state['score'] > self.max_value:
            self.max_value = state['score']
        # first add gravity to this state object
        # if we jumped last time, figure out what gravity is
        if self.last_action == 1:
            self.gravity = state['monkey']['top'] - self.last_state['monkey']['top']
        state['gravity'] = self.gravity

        if self.is_first:
            self.is_first = False
            self.last_state = state
            self.last_action = int(npr.random() > 0.5)
            return self.last_action

        # get q(s,a) values for current state
        no_jump_q_val, jump_q_val = self.get_q_values(state)

        # get epsilon decision
        rand_val = npr.random()
        threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.actions_taken / EPSILON_DECAY)
        print("what is threshold", threshold)
        self.actions_taken += 1

        # epsilon greedy decision here, get next action and its value
        new_action = None
        new_action_value = None
        if rand_val > threshold:
            # use our current q values
            if jump_q_val > no_jump_q_val:
                new_action = 1
            else:
                new_action = 0
        else:
            # explore randomly
            new_action = int(npr.random() > 0.5)
        # get the value of best
        if jump_q_val > no_jump_q_val:
            new_action_value = jump_q_val
        else:
            new_action_value = no_jump_q_val

        # infor from previous state
        tree_dist, monkey_top_to_tree_top, monkey_bot_to_tree_bot, monkey_vel_idx, gravity, vel_pos, monkey_top_below, monkey_bot_above = self.get_q_value_indexes(state)
        # value from last state
        last_action_value = self.Q_values[self.last_action][tree_dist][monkey_top_to_tree_top][monkey_bot_to_tree_bot][monkey_vel_idx][gravity][vel_pos][monkey_top_below][monkey_bot_above]
        # You might do some learning here based on the current state and the last state.
        # update q table
        self.Q_values[self.last_action][tree_dist][monkey_top_to_tree_top][monkey_bot_to_tree_bot][monkey_vel_idx][gravity][vel_pos][monkey_top_below][monkey_bot_above] = \
            last_action_value + (LEARNING_RATE * (self.last_reward + (GAMMA * new_action_value) - last_action_value))

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

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
        run_games(agent, hist, 1000, 0)

        print(agent.max_value)

        # Save history.
        np.save('hist',np.array(hist))


