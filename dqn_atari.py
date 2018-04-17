from __future__ import print_function, division

import torch
import torch.optim as optim
import numpy as np
from PIL import Image

from networks.simplenet import AtariConvNet
from agents.simul import DQNSimulator
from agents.dqn import DQN
from utils.env_wrapper import AtariEnv
from utils.memory import ReplayMemory


# Training parameters
TOTAL_STEPS = 10000000
TARGET_UPDATE_STEPS = 10000
HISTORY_LENGTH = 4
EXPLORATION_STEPS = 50000

# Data parameters
SCREEN_SIZE = (84, 84)


def before_replay_process(screen):
    s = Image.fromarray(screen)
    s = s.resize(SCREEN_SIZE).convert('L')
    s = np.array(s)
    return np.asarray(s, dtype=np.uint8)


def process(state):
    if state is None:
        return None
    assert state.dtype == np.uint8
    assert state.shape == (4, 84, 84)
    return state.astype(np.float32) / 255.0


def make_cuda(model):
    if torch.cuda.is_available():
        model.cuda()


def main():
    # set up environment
    env = AtariEnv('PongNoFrameskip-v4', noop_max=30, skip=4, seed=123456)

    # set up network model
    in_channels = HISTORY_LENGTH
    num_actions = env.num_actions

    model = AtariConvNet(in_channels, num_actions)
    make_cuda(model)

    target_model = AtariConvNet(in_channels, num_actions)
    make_cuda(target_model)

    optimizer = optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    # set up deep q network
    dqn = DQN(model, optimizer, target_model=target_model, gamma = 0.99,
            double_q_learning=False, eps_start=1.0, eps_end=0.1, 
            eps_decay=1000000)
    replay = ReplayMemory(200000, history_length=HISTORY_LENGTH)

    simul = DQNSimulator(dqn, env, replay)
    simul.train(TOTAL_STEPS, 
                target_update_steps=TARGET_UPDATE_STEPS, 
                batch_size=32,
                exploration_steps=EXPLORATION_STEPS, 
                before_replay_process_func=before_replay_process,
                process_func=process)

    env.close()

if __name__ == '__main__':
    main()
