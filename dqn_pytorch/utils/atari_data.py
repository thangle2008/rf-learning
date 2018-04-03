import gym

import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize


class AtariEnvWrapper(object):

    def __init__(self, env_name, num_frames=4, screen_size=(84, 84), 
                 seed = None):
        
        self.env = gym.make(env_name)
        if seed:
            self.env.seed(seed)
        # state is now a stack of frame (top is the newest)
        self.state = np.zeros(screen_size + (num_frames,), dtype=np.float32)
        self.frame = 0
    

    def preprocessing(screen):
        """Preprocess a screen image."""

        # remove rgb channel and resize to 84x84
        screen = rgb2grey(screen)
        screen = resize(screen, (ATARI_SIZE, ATARI_SIZE))
        # normalize
        screen = screen.astype(np.float32) / 255.0
        return screen

    
    def reset(self):
        """Reset the environment and return a starting state."""

        first_screen = preprocessing(env.reset())
        for i in range(self.num_frames):
            self.state[i] = first_screen
        return np.asarray(self.state, dtype=np.float32)


    def step(self, action):
        """Perform an action and have the environment return a new state, reward, and status."""

        next_state, reward, done, _ = env.step(action)
        # update frame
        self.state[self.frame] = next_state
        self.frame = (self.frame + 1) % self.num_frames
        return np.asarray(self.state, dtype=np.float32), reward, done
