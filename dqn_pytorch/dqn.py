import random
from collections import deque, namedtuple

import torch
from torch.autograd import Variable


Transition = namedtuple('Transition', 
                        ['state', 'action', 'next_state', 'reward']) 

ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else \
             torch.ByteTensor


class DQN(object):

    def __init__(self, model, replay_size=10000, batch_size=32):
        """Initialize a DQN from a network model."""

        self.model = model
        self.replay_memory = deque(maxlen = replay_size)

    
    def update(self):
        """Sample a batch of transitions from replay memory and train the network on it."""

        transitions = random.sample(self.replay_memory, batch_size)
        batch = Transition(*zip(*transitions))

        # get non-final states
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
        
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)

        
