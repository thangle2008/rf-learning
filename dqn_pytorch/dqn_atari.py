from __future__ import print_function, division

import torch
import torch.optim as optim

from networks.simplenet import AtariConvNet
from agents.dqn import DQN
from utils.atari_data import AtariEnvWrapper
from core import simul

TOTAL_STEPS = 10000000
TARGET_UPDATE_STEPS = 10000


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor


def make_cuda(model):
    if torch.cuda.is_available():
        model.cuda()


def main():
    # set up environment
    env = AtariEnvWrapper('PongNoFrameskip-v4', noop_max=30, skip=4,
                          seed=123456)

    # set up network model
    in_channels = env.num_frames
    num_actions = env.num_actions

    model = AtariConvNet(in_channels, num_actions)
    make_cuda(model)

    target_model = AtariConvNet(in_channels, num_actions)
    make_cuda(target_model)

    optimizer = optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    dqn = DQN(model, optimizer, target_model=target_model, gamma = 0.99,
            double_q_learning=False, eps_start=1.0, eps_end=0.1, 
            eps_decay=1000000, replay_size=1000000)

    simul.train(dqn, env, num_steps=TOTAL_STEPS, 
                exploration_steps=50000,
                target_update_steps=TARGET_UPDATE_STEPS,
                save_path='./trained_models/model.pkl')

    env.close()

if __name__ == '__main__':
    main()
