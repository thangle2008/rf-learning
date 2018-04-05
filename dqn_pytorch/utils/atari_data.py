import gym

import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize


class AtariEnvWrapper(object):

    def __init__(self, env_name, num_frames=4, screen_size=(84, 84), 
                 seed = None):
        
        self.env = gym.make(env_name)
        self.screen_size = screen_size
        self.num_actions = self.env.action_space.n
        self.num_frames = num_frames
        if seed:
            self.env.seed(seed)
        # state is now a stack of frame (top is the newest)
        self.state = np.zeros((num_frames,) + screen_size, dtype=np.float32)
        self.frame = 0
    

    def preprocessing(self, screen):
        """Preprocess a screen image."""

        # remove rgb channel and resize to 84x84
        screen = rgb2grey(screen)
        screen = resize(screen, self.screen_size)
        # normalize
        screen = screen.astype(np.float32) / 255.0
        return screen

    
    def reset(self):
        """Reset the environment and return a starting state."""

        first_screen = self.preprocessing(self.env.reset())
        for i in range(self.num_frames):
            self.state[i] = first_screen
        return np.asarray(self.state, dtype=np.float32)


    def step(self, action):
        """Perform an action and have the environment return a new state, reward, and status."""

        next_state, reward, done, info = self.env.step(action)
        # update frame
        self.state[self.frame] = self.preprocessing(next_state)
        self.frame = (self.frame + 1) % self.num_frames
        return np.asarray(self.state, dtype=np.float32), reward, done, info
