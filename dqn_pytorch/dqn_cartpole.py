from __future__ import print_function, division

import argparse
from collections import deque
import gym

import torch
import torch.optim as optim

from simplenet import SimpleANN
from dqn import Transition, DQN


NUM_EPISODES = 1500
TIMES_SOLVED = 200
DOUBLE_Q = True


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor



def numpy_to_tensor(s):
    return torch.from_numpy(s).type(FloatTensor).unsqueeze(0)


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

    target_model=None
    if DOUBLE_Q:
        target_model = SimpleANN(input_size, output_size)
        make_cuda(target_model)

    optimizer = optim.Adam(model.parameters())

    dqn = DQN(model, optimizer, target_model=target_model, 
        eps_start=1.0, eps_end=0.01, eps_decay=30000, replay_size=2000)

    time = 0
    last_scores = deque(maxlen=TIMES_SOLVED)
    best_avg_score = 0.0

    for e in range(NUM_EPISODES):
        if DOUBLE_Q:
            dqn.update_target_network()

        state = env.reset()
        state = numpy_to_tensor(state)

        done = False

        last_time = time
        while not done:
            #env.render()

            # pick an action and move onto next state
            action = dqn.select_action(state, time)
            next_state, reward, done, _ = env.step(action[0, 0])

            reward = reward if not done else -10
            reward = FloatTensor([reward])

            time += 1
            
            next_state = None if done else numpy_to_tensor(next_state)

            # store transition in replay memory
            dqn.remember(state, action, next_state, reward)
            
            # optimize if there is enough in replay memory
            dqn.update()
            state = next_state

        score = time - last_time
        last_scores.append(score)
        print(score)

        if len(last_scores) == TIMES_SOLVED:
            avg_score = sum(last_scores) / len(last_scores)
            if best_avg_score < avg_score:
                best_avg_score = avg_score
                dqn.save_model("./saved_models/dqn_cartpole.pkl")


    env.close()


if __name__ == '__main__':
    main()
