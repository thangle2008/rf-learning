from __future__ import print_function, division

import torch
import torch.optim as optim

from networks.simplenet import SimpleANN
from core.simul import DQNSimulator
from agents.dqn import DQN
from utils.env_wrapper import BasicEnv
from utils.memory import ReplayMemory


TOTAL_STEPS = 10000
TARGET_UPDATE_STEPS = 300
EXPLORATION_STEPS = 2000


def make_cuda(model):
    if torch.cuda.is_available():
        model.cuda()


def main():
    # set up environment
    env = BasicEnv('CartPole-v1', seed=123456)

    # set up network model
    input_size = 4
    output_size = env.num_actions

    model = SimpleANN(input_size, output_size)
    make_cuda(model)

    target_model = SimpleANN(input_size, output_size)
    make_cuda(target_model)

    optimizer = optim.Adam(model.parameters())

    dqn = DQN(model, optimizer, target_model=target_model, gamma = 0.99,
            double_q_learning=True, eps_start=1.0, eps_end=0.05, 
            eps_decay=10000)
    replay = ReplayMemory(10000, history_length=1)

    simul = DQNSimulator(dqn, env, replay)
    simul.train(TOTAL_STEPS, 
                target_update_steps=TARGET_UPDATE_STEPS, 
                batch_size=32,
                exploration_steps=EXPLORATION_STEPS, 
                save_path='./trained_models/',
                save_steps=200)
    simul.test(5, batch_size=32)

    env.close()


if __name__ == '__main__':
    main()
