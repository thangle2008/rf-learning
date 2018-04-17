from __future__ import division, print_function

import math
import copy
from collections import deque, namedtuple
import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F


ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else \
             torch.ByteTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else \
             torch.LongTensor


class DQN(object):

    def __init__(self, model, optimizer, target_model = None, double_q_learning = False, 
                 gamma=0.95, eps_start=0.9, eps_end=0.05, eps_decay=200):
        """Initialize a DQN from a network model."""

        self.model = model

        # initalize target Q function
        self.target_model = target_model
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # some flags
        self.double_q_learning = double_q_learning

        # optimizer
        self.optimizer = optimizer

        # hyperparameters
        self.gamma = gamma

        # eps greedy policy
        self.eps_current = self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_steps = 0

    
    def update(self, batch):
        """Train the agent using a batch of transitions."""
        batch_size = len(batch.state)

        # convert to tensors
        state_batch = tuple(FloatTensor(s).unsqueeze(0) for s in batch.state)
        action_batch = tuple(LongTensor([a]).unsqueeze(0) for a in batch.action)
        reward_batch = tuple(FloatTensor([r]) for r in batch.reward)

        # extract contents from this batch
        state_batch = Variable(torch.cat(state_batch))
        action_batch = Variable(torch.cat(action_batch))
        reward_batch = Variable(torch.cat(reward_batch))

        # get non-final next states
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
        
        non_final_next_states = \
            Variable(torch.cat([FloatTensor(s).unsqueeze(0) for s in 
                                batch.next_state if s is not None]), 
                     volatile=True)

        # compute Q(s_t) and use that to gather Q(s_t, a) by picking
        # the correct actions
        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = Variable(
            torch.zeros(batch_size).type(FloatTensor))

        if self.double_q_learning: 
            # use online network to find best actions
            best_actions = self.model(non_final_next_states).max(1)[1].unsqueeze(1)
            # use target network to determine the values
            target_state_values = self.target_model(non_final_next_states)
            next_state_values[non_final_mask] = \
                target_state_values.gather(1, best_actions).squeeze(1)
        else:
            next_state_values[non_final_mask] = \
                self.target_model(non_final_next_states).max(1)[0]

        expected_state_action_values = next_state_values * self.gamma + \
                                                                reward_batch
        # undo volatiability
        expected_state_action_values = Variable(expected_state_action_values.data)

        # compute loss based on Bellman's equation
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        """Update target network's weights."""
        self.target_model.load_state_dict(self.model.state_dict())


    def select_action(self, state, deterministic=False):
        """Choose most optimal action for a given state."""

        self.eps_current = self.eps_start - \
            float(self.eps_steps) / self.eps_decay * (self.eps_start - self.eps_end)
        self.eps_current = max(self.eps_current, self.eps_end)
        self.eps_steps += 1

        # convert state to tensor
        state = FloatTensor(state).unsqueeze(0)

        if deterministic or random.random() >= self.eps_current:
            q_s = self.model(Variable(state, volatile=True).type(FloatTensor))
            return q_s.data.max(1)[1][0]
        else:
            return self.random_action()


    def random_action(self):
        """Random an action."""

        return random.randrange(0, self.model.num_actions)


    def save_model(self, path):
        """Save the parameters of the model."""

        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        """Load parameters to the model."""

        self.model.load_state_dict(torch.load(path))
