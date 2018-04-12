import gym

import numpy as np
import random

from PIL import Image

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

    def __init__(self, env_name, screen_size=(84, 84), num_frames=4, 
                 noop_max=30, skip=4, seed=None):
        
        super(AtariEnv, self).__init__(env_name, seed)
        self.screen_size = screen_size
        self.num_actions = self.env.action_space.n
        self.num_frames = num_frames
        self.noop_max = noop_max
        self.skip = skip
        # state is now a stack of frames (top is the newest)
        self.state_buffer = []
    
    
    def preprocessing(self, screen):
        s = Image.fromarray(screen)
        s = s.resize(self.screen_size).convert('L')
        s = np.array(s)
        return np.asarray([s], dtype=np.uint8)


    def reset(self):
        """Reset the environment and return a starting state."""

        screen = self.env.reset()

        # perform no-op
        for _ in range(random.randrange(0, self.noop_max + 1)):
            screen = self.env.step(0)[0]

        self.state_buffer = [self.preprocessing(screen)] * self.num_frames
        return np.vstack(self.state_buffer)


    def step(self, action):
        """Perform an action and have the environment return a new state, reward, and status."""

        total_reward = 0.0
        for _ in range(self.skip):
            screen, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # update screen stack
        self.state_buffer.pop(0)
        self.state_buffer.append(self.preprocessing(screen))
        state = np.vstack(self.state_buffer)

        return state, total_reward, done, info
