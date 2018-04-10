import gym

import numpy as np

from skimage.color import rgb2grey
from skimage.transform import resize


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


class AtariEnvWrapper(EnvWrapper):

    def __init__(self, env_name, num_frames=4, screen_size=(84, 84), 
                 noop_max=30, skip=4, seed=None):
        
        super(AtariEnvWrapper, self).__init__(env_name, seed)
        self.screen_size = screen_size
        self.num_actions = self.env.action_space.n
        self.num_frames = num_frames
        self.noop_max = noop_max
        self.skip = skip
        # state is now a stack of frames (top is the newest)
        self.state = np.zeros((num_frames,) + screen_size, dtype=np.float32)
        self.frame = 0
    

    def preprocessing(self, screen):
        """Preprocess a screen image."""

        # remove rgb channel and resize to 84x84
        screen = rgb2grey(screen)
        screen = resize(screen, self.screen_size)
        return screen.astype(np.float32)

    
    def reset(self):
        """Reset the environment and return a starting state."""

        screen = self.env.reset()

        # perform no-op
        for _ in range(np.random.randint(0, self.noop_max)):
            screen = self.env.step(0)[0]

        screen = self.preprocessing(screen)
        for i in range(self.num_frames):
            self.state[i] = screen
        return np.asarray(self.state, dtype=np.float32)


    def step(self, action):
        """Perform an action and have the environment return a new state, reward, and status."""

        total_reward = 0.0
        for _ in range(self.skip):
            screen, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # update frame
        self.state[self.frame] = self.preprocessing(screen)
        self.frame = (self.frame + 1) % self.num_frames
        return np.asarray(self.state, dtype=np.float32), total_reward, done, info
