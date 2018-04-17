import gym

import numpy as np
import random

class EnvWrapper(object):

    def __init__(self, env_name, seed=None):

        self.env = gym.make(env_name)
        if seed:
            self.env.seed(seed)


    def render(self):
        """Draw current screen image."""

        self.env.render()


    def close(self):
        """Close the environment."""

        self.env.close()


class AtariEnv(EnvWrapper):

    def __init__(self, env_name, noop_max=30, skip=4, seed=None):
        
        super(AtariEnv, self).__init__(env_name, seed)
        self.num_actions = self.env.action_space.n
        self.noop_max = noop_max
        self.skip = skip
    
    
    def reset(self):
        """Reset the environment and return a starting state."""

        screen = self.env.reset()

        # perform no-op to introduce stochasticity 
        for _ in range(random.randrange(0, self.noop_max + 1)):
            screen = self.env.step(0)[0]

        return screen


    def step(self, action):
        """Perform an action and have the environment return a new state, reward, and status."""

        total_reward = 0.0
        for _ in range(self.skip):
            screen, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return screen, total_reward, done, info
