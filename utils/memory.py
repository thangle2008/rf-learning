from collections import deque, namedtuple
import random
import copy

import numpy as np


# in a given state, perform action, get reward and move to next_state
# note that if next_state is a final state, then it is None
Transition = namedtuple('Transition', 
                        ['state', 'action', 'next_state', 'reward']) 


class ArrayDeque(object):
    """Array implementation of double-ended queue."""

    def __init__(self, maxlen):

        self.data = []
        self.maxlen = maxlen
        self.start = 0
        self.length = 0


    def append(self, v):

        if self.length < self.maxlen:
            self.length += 1
            self.data.append(v)
        elif self.length == self.maxlen:
            # store the value at the end (old start) and update start position
            self.data[self.start] = v
            self.start = (self.start + 1) % self.maxlen

    
    def __len__(self):

        return self.length


    def __getitem__(self, idx):

        return self.data[(self.start + idx) % self.maxlen]


class ReplayMemory(object):

    def __init__(self, maxlen, history_length=4):

        self.history_length = history_length
        self.states = ArrayDeque(maxlen)
        self.actions = ArrayDeque(maxlen)
        self.rewards = ArrayDeque(maxlen)


    def remember(self, state, action, reward):
        """Store a new observation. (TODO: should we have a terminal flag here?)."""

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def size(self):

        return len(self.states)
    
    
    def _get_states_from_idx(self, idx, current_state):

        res = [current_state]

        for i in range(self.history_length - 1):
            next_idx = idx - i
            if next_idx < 0 or next_idx >= len(self.states):
                break

            if self.states[next_idx] is not None:
                res.insert(0, self.states[next_idx])
            else:
                break

        # add dummy states if the stack has not been filled
        while len(res) < self.history_length:
            res.insert(0, self._get_dummy_state(current_state)) 

        for i in range(self.history_length):
            res[i] = self._wrap_state(res[i])

        res = np.vstack(res)
        assert res.dtype == current_state.dtype
        return res

    
    def _wrap_state(self, state):

        return np.asarray([state], dtype=state.dtype)


    def _get_dummy_state(self, current_state):

        return np.zeros(current_state.shape, dtype=current_state.dtype)


    def get_recent_states(self, current_state):
        """Stack the most recent states with the current_state."""

        return self._get_states_from_idx(len(self.states)-1, current_state)

    
    def sample(self, batch_size):
        """Sample a batch of state stacks."""

        # Sample the indexes for the current states (last state in each stack
        # frame). We do not want the first few states since we want to fill
        # the stack. We also do not want the last state since we do not know 
        # what the next state will be yet.
        if len(self.states) - (self.history_length - 1) - 1 < batch_size:
            return None

        transitions = []

        batch_idxs = random.sample(
            range(self.history_length - 1, len(self.states) - 1), batch_size)
        
        for idx in batch_idxs:
            # get the current state, if it is a final state, resample another
            # one (possibly duplicate)
            while self.states[idx] is None:
                idx = random.randrange(self.history_length - 1,
                                       len(self.states) - 1)
            state = self.states[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            
            # fill the stack with previous states
            s_stack = self._get_states_from_idx(idx-1, state)

            # get next stack
            next_stack = None
            if self.states[idx+1] is not None:
                next_stack = np.zeros(s_stack.shape, dtype=s_stack.dtype)
                next_stack[0:self.history_length-1] = s_stack[1:]
                next_stack[-1] = self.states[idx+1]
                assert np.all(next_stack[0:self.history_length-1] == s_stack[1:])

            assert len(s_stack) == self.history_length
            assert next_stack is None or len(next_stack) == self.history_length

            trans = Transition(s_stack, action, next_stack, reward)
            transitions.append(trans)


        assert len(transitions) == batch_size
        
        # separate state, action, reward, next_state batch
        batch = Transition(*zip(*transitions))

        return batch
