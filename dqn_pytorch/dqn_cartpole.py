from __future__ import print_function, division

import argparse
from collections import deque
import gym

import torch
import torch.optim as optim

from simplenet import SimpleANN
from dqn import Transition, DQN

TOTAL_STEPS = 80000
TARGET_UPDATE_STEPS = 300

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor


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

    # initialize state
    current_reward = 0.0
    state = env.reset()
    episode = 1

    for t in range(TOTAL_STEPS):
        # pick an action
        action = dqn.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done: 
            next_state = None

        current_reward += reward

        # store transition in replay memory
        dqn.remember(state, action, next_state, reward)
        
        # optimize if there is enough in replay memory
        dqn.update()

        # update the target network after fixed steps
        if t % TARGET_UPDATE_STEPS == 0:
            dqn.update_target_network()

        # move onto next state
        state = next_state
        if done: 
            print("Episode {} at t = {}/{}: reward = {}, eps = {:.3f}".format(
                episode, t, TOTAL_STEPS, current_reward, dqn.eps_current))
            state = env.reset()
            current_reward = 0.0
            episode += 1

    # test the model in 5 episodes
    for e in range(5):
        state = env.reset()
        done = False
        current_reward = 0
        while not done:
            env.render()
            action = dqn.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            current_reward += reward
            state = next_state

        print("Episode {}/5: reward = {}".format(e + 1, current_reward))

    env.close()

if __name__ == '__main__':
    main()
