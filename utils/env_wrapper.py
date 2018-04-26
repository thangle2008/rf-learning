import gym

import numpy as np
import random
import time


class BasicEnv(object):

    def __init__(self, env_name, seed=None):

        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        if seed:
            self.env.seed(seed)


    def render(self):
        """Draw current screen image."""

        self.env.render()


    def close(self):
        """Close the environment."""

        self.env.close()


    def step(self, action, test=False):
        if test:
            self.render()
        state, reward, done, info = self.env.step(action)
        return state, np.sign(reward), done, info


    def reset(self):

        return self.env.reset()


class AtariEnv(BasicEnv):

    def __init__(self, env_name, noop_max=0, skip=1, seed=None):
        
        super(AtariEnv, self).__init__(env_name, seed)
        self.noop_max = noop_max
        self.skip = skip
    
    
    def set_noop_max(self, val):
        
        self.noop_max = val

    
    def set_frame_skip(self, val):
        
        self.skip = val


    def reset(self):
        """Reset the environment and return a starting state."""

        screen = self.env.reset()

        # perform no-op to introduce stochasticity 
        for _ in range(random.randrange(0, self.noop_max + 1)):
            screen, _, done, _ = self.env.step(0)
            if done:
                screen = self.env.reset()

        return screen


    def step(self, action, test=False):
        """Perform an action and have the environment return a new state, reward, and status."""

        total_reward = 0.0
        for _ in range(self.skip):
            if test:
                self.env.render()
            #    time.sleep(0.005)
            screen, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return screen, np.sign(total_reward), done, info
