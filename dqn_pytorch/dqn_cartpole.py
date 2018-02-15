from __future__ import print_function

import gym

import torch
import torch.optim as optim

from simplenet import SimpleANN
from dqn import Transition, DQN


NUM_EPISODES = 5000 


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor


def numpy_to_tensor(s):
    return torch.from_numpy(s).type(FloatTensor).unsqueeze(0)


def main():
    # set up environment
    env = gym.make('CartPole-v0')

    # set up network model
    input_size = env.observation_space.shape[0] # should be 4
    output_size = env.action_space.shape[0]     # should be 2

    model = SimpleANN(input_size, output_size)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters())
    dqn = DQN(model, optimizer, eps_start=1.0, eps_end=0.01,
              eps_decay=10000)

    time = 0

    for e in range(NUM_EPISODES):
        state = env.reset()
        state = numpy_to_tensor(state)

        done = False

        last_time = time
        while not done:
            env.render()

            # pick an action and move onto next state
            action = dqn.select_action(state, time)
            next_state, reward, done, _ = env.step(action[0, 0])
            reward = FloatTensor([reward])

            time += 1
            
            next_state = None if done else numpy_to_tensor(next_state)

            # store transition in replay memory
            dqn.remember(state, action, next_state, reward)
            
            # optimize if there is enough in replay memory
            dqn.update()
            state = next_state

        score = time - last_time
        print("Episode {}/{} has score = {}".format(e + 1, NUM_EPISODES, score))

    env.render(close=True)
    env.close()


if __name__ == '__main__':
    main()
