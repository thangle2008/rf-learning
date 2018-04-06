from __future__ import print_function, division

from collections import deque
import gym

import torch
import torch.optim as optim

from networks.simplenet import SimpleANN
from agents.dqn import Transition, DQN
from core import simul

TOTAL_STEPS = 80000
TARGET_UPDATE_STEPS = 300


def make_cuda(model):
    if torch.cuda.is_available():
        model.cuda()


def main():
    # set up environment
    env = gym.make('CartPole-v1')
    env.seed(123456)

    # set up network model
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n 

    model = SimpleANN(input_size, output_size)
    make_cuda(model)

    target_model = SimpleANN(input_size, output_size)
    make_cuda(target_model)

    optimizer = optim.Adam(model.parameters())

    dqn = DQN(model, optimizer, target_model=target_model, gamma = 0.99,
            double_q_learning=True, eps_start=1.0, eps_end=0.05, 
            eps_decay=10000, replay_size=10000)

    simul.train(dqn, env, num_steps=TOTAL_STEPS, target_update_steps=TARGET_UPDATE_STEPS) 
    simul.test(dqn, env, num_episodes=5, verbose=1)

    env.close()


if __name__ == '__main__':
    main()
