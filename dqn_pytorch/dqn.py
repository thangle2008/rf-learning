from __future__ import division, print_function

import random
import math
from collections import deque, namedtuple

import torch
from torch.autograd import Variable
import torch.nn.functional as F


Transition = namedtuple('Transition', 
                        ['state', 'action', 'next_state', 'reward']) 


ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else \
             torch.ByteTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else \
              torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else \
             torch.LongTensor



class DQN(object):

    def __init__(self, model, optimizer, target_model=None, 
                 replay_size=1000, batch_size=32,
                 gamma=0.95, eps_start=0.9, eps_end=0.05, eps_decay=200):
        """Initialize a DQN from a network model."""

        self.model = model
        self.target_model = target_model # for double q learning 

        self.optimizer = optimizer
        self.replay_memory = deque(maxlen = replay_size)
        self.batch_size = batch_size

        # hyperparameters
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    
    def update(self):
        """Sample a batch of transitions from replay memory and train the network on it."""

        if (len(self.replay_memory) < self.batch_size):
            return

        transitions = random.sample(self.replay_memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        # extract contents from this batch
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # compute Q(s_t) and use that to gather Q(s_t, a) by picking
        # the correct actions
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # get non-final states
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
        
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)

        next_state_values = Variable(
                torch.zeros(self.batch_size).type(FloatTensor))

        if self.target_model is not None: # double q learning
            # use online network to find best actions
            best_actions = self.model(non_final_next_states).max(1)[1].unsqueeze(1)
            # use target network to determine the values
            target_state_values = self.target_model(non_final_next_states)
            next_state_values[non_final_mask] = \
                target_state_values.gather(1, best_actions)
        else:
            next_state_values[non_final_mask] = \
                self.model(non_final_next_states).max(1)[0]


        next_state_values.volatile = False
        
        expected_state_action_values = next_state_values * self.gamma + \
                                                                reward_batch

        # compute loss based on Bellman's equation
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def update_target_network(self):
        """Update target network's weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        

    def remember(self, *args):
        """Store a transition in the replay memory."""

        self.replay_memory.append(Transition(*args))


    def select_action(self, state, time):
        """Choose most optimal action for a given state."""

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1 * time / self.eps_decay)

        if random.random() > eps_threshold:
            q_s = self.model(Variable(state, volatile=True).type(FloatTensor))
            return q_s.data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.model.output_size)]])


    def save_model(self, path):
        """Save the parameters of the model."""

        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        """Load parameters to the model."""

        self.model.load_state_dict(torch.load(path))
