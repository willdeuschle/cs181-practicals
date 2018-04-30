# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

BIN_SIZE_HEIGHT = 100
BIN_SIZE_WIDTH = 150
NUM_GRAVITY_BINS = 2
GAMMA = 0.9
LEARNING_RATE = 0.1
EPSILON_START = 0.95
EPSILON_END = 0.001
EPSILON_DECAY = 1200
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
             int(SCREEN_WIDTH / BIN_SIZE_WIDTH), # tree dist (horizontal distance)
             int(SCREEN_HEIGHT / BIN_SIZE_HEIGHT), # tree  - monkey  (vertical distance)
             #int(SCREEN_HEIGHT / BIN_SIZE_HEIGHT + 1), # monkey bottom
             NUM_GRAVITY_BINS, # gravity
             #NUM_VEL_BINS,
            2 # velocity pos/neg
        ))

        # if monkey is very low, jump
        #self.Q_values[:,2,:,0,:,:] = 1
        #print(self.Q_values)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.is_first = True
        print(self.gravity)
        self.gravity = 0

    def get_q_value_indexes(self, state):
        # get this state in terms of Q
        tree_dist = int(state['tree']['dist'] / BIN_SIZE_WIDTH)
        v_dist = int((state['tree']['top'] - state['monkey']['top'])/ BIN_SIZE_HEIGHT)
        #print('tree', state['tree']['top'])
        #print('monkey', state['monkey']['top'])
        #print('vdist', v_dist)
        #tree_bot = int(state['tree']['bot'] / BIN_SIZE_HEIGHT)
        #monkey_top = int(state['monkey']['top'] / BIN_SIZE_HEIGHT)
        #monkey_bot = int(state['monkey']['bot'] / BIN_SIZE_HEIGHT)

        monkey_vel = state['monkey']['vel']

        # 1 for up, 0 for down
        vel_pos = int(monkey_vel >= 0)
        #print('our gravity', gravity)

        return (tree_dist, v_dist, self.gravity, vel_pos)

    def get_q_values(self, state):
        tree_dist, v_dist, gravity, vel_pos = self.get_q_value_indexes(state)

        no_jump_q_val = self.Q_values[0][tree_dist][v_dist][gravity][vel_pos]
        jump_q_val = self.Q_values[1][tree_dist][v_dist][gravity][vel_pos]
        
        #print("yes", jump_q_val, "no", no_jump_q_val)
        return (no_jump_q_val, jump_q_val)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        if state['score'] > self.max_value:
            self.max_value = state['score']
        # first add gravity to this state object
        # if falling, compute gravity

        if self.is_first:
            self.is_first = False
            self.last_state = state
            self.last_action = 0
            return self.last_action

        vel_diff = None
        if self.last_action == 0:
            vel_diff =  self.last_state['monkey']['vel'] - state['monkey']['vel']

        if (vel_diff):
            #print('velldiff', vel_diff)
            if (vel_diff >2):
                self.gravity = 1
            else:
                self.gravity = 0
        #state['gravity'] = self.gravity

        # get q(s,a) values for current state
        no_jump_q_val, jump_q_val = self.get_q_values(state)

        # get epsilon decision
        rand_val = npr.random()
        threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.actions_taken / EPSILON_DECAY)
        #print("what is threshold", threshold)
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
            new_action = int(npr.random() > 0.7)

        # get the value of best
        if jump_q_val > no_jump_q_val:
            new_action_value = jump_q_val
        else:
            new_action_value = no_jump_q_val

        # infor from previous state
        tree_dist, v_dist, gravity, vel_pos = self.get_q_value_indexes(self.last_state)
        # value from last state
        last_action_value = self.Q_values[self.last_action][tree_dist][v_dist][gravity][vel_pos]
        # You might do some learning here based on the current state and the last state.
        # update q table
        self.Q_values[self.last_action][tree_dist][v_dist][gravity][vel_pos] = \
            last_action_value + (LEARNING_RATE * (self.last_reward + (GAMMA * new_action_value) - last_action_value))

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
        print('G:', swing.gravity, "score:", swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, 500, 0)

        print('max score', agent.max_value)
        print('mean score', np.mean(hist))

        # Save history.
        np.save('hist',np.array(hist))
