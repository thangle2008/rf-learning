from __future__ import print_function, division
import argparse

import torch
import torch.optim as optim
import numpy as np
from PIL import Image

from networks.simplenet import AtariConvNet
from core.simul import DQNSimulator
from agents.dqn import DQN
from utils.env_wrapper import AtariEnv
from utils.memory import ReplayMemory


parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test_model_path')
parser.add_argument('--env_name', dest='env_name', default="PongNoFrameskip-v4")

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


def after_replay_process(state):
    if state is None:
        return None
    assert state.dtype == np.uint8
    assert state.shape == (HISTORY_LENGTH,) + SCREEN_SIZE
    return state.astype(np.float32) / 255.0


def make_cuda(model):
    if torch.cuda.is_available():
        model.cuda()


def main(args):
    # set up environment
    env = AtariEnv(args.env_name, skip=4, seed=123456)

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
            eps_decay=500000)

    if args.test_model_path:
        print("Testing...")
        # load model
        dqn.load_model(args.test_model_path)
        # only need enough to stack frames
        replay = ReplayMemory(HISTORY_LENGTH, history_length=HISTORY_LENGTH)
        simul = DQNSimulator(dqn, env, replay,
                             before_replay_process_func=before_replay_process,
                             after_replay_process_func=after_replay_process)
        simul.test(5, batch_size=32)
    else:
        print("Training...")
        # make env stochastic
        env.set_noop_max(30)
        replay = ReplayMemory(500000, history_length=HISTORY_LENGTH)
        simul = DQNSimulator(dqn, env, replay,
                             before_replay_process_func=before_replay_process,
                             after_replay_process_func=after_replay_process)
        simul.train(TOTAL_STEPS, 
                    target_update_steps=TARGET_UPDATE_STEPS, 
                    batch_size=32,
                    exploration_steps=EXPLORATION_STEPS, 
                    save_path='./trained_models/',
                    save_steps=50000)

    env.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
