import numpy as np
import os

from utils.memory import Transition


class BaseSimulator(object):

    def __init__(self, agent, env):

        self.agent = agent
        self.env = env


class DQNSimulator(BaseSimulator):
    """Simulator for training DQN."""

    def __init__(self, agent, env, replay, before_replay_process_func=None,
                 after_replay_process_func=None):

        super(DQNSimulator, self).__init__(agent, env)
        self.replay = replay
        self.before_replay_process_func = before_replay_process_func
        self.after_replay_process_func = after_replay_process_func


    def _one_step(self, state, exploration_steps, test=False):
        if self.before_replay_process_func:
            state = self.before_replay_process_func(state)

        # stack current state with previous states to choose an action
        s_stack = self.replay.get_recent_states(state)

        # pick an action
        if self.after_replay_process_func:
            s_stack = self.after_replay_process_func(s_stack)

        if exploration_steps == 0:
            action = self.agent.select_action(s_stack, deterministic=test)
        else: 
            action = self.agent.random_action()
        next_state, reward, done, _ = self.env.step(action, test)

        # store recent observation in replay memory
        self.replay.remember(state, action, reward)

        # end episode
        if done: 
            # remember this last state with dummy action and reward
            self.replay.remember(None, -1, -1)

        return next_state, reward, done


    def train(self, num_steps, target_update_steps=200, exploration_steps=0, 
              batch_size=32, save_path=None, save_steps=1):

        # initialize state
        total_reward = 0.0
        state = self.env.reset()
        episode = 1

        last_t = 1
        for t in range(1, num_steps + 1):
            next_state, reward, done = self._one_step(state, exploration_steps)
            total_reward += reward
            
            # move onto next state
            state = next_state

            if exploration_steps > 0:
                exploration_steps -= 1
                if exploration_steps == 0:
                    print("Done exploring")
            else:
                # train network if there are enough samples
                # in replay memory
                batch = self.replay.sample(batch_size)
                if self.after_replay_process_func:
                    new_state = tuple(map(self.after_replay_process_func, 
                                          batch.state))
                    new_next_state = tuple(map(self.after_replay_process_func, 
                                               batch.next_state))
                    batch = Transition(new_state, batch.action, new_next_state, 
                                       batch.reward)
                self.agent.update(batch)

                # update the target network after fixed steps
                if t % target_update_steps == 0:
                    self.agent.update_target_network()

            # output every episode and reset state
            if done:
                print(("Episode {} at t = {}/{}: reward = {}, eps = {:.3f}, "
                       "steps = {}").format(episode, t, num_steps, total_reward,
                                            self.agent.eps_current,
                                            t - last_t + 1))
                state = self.env.reset()
                total_reward = 0.0
                last_t = t
                episode += 1

            # save model periodically
            if save_path and t % save_steps == 0:
                self.agent.save_model(os.path.join(save_path, 'model.pkl'))
                self.agent.save_optim(os.path.join(save_path, 'optim.pkl'))


    def test(self, num_episodes, batch_size=32):
        """Test the agent in several episodes."""

        for e in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            t = 0
            while True:
                t += 1
                next_state, reward, done = self._one_step(state, 0, True)
                state = next_state
                total_reward += reward
                if done:
                    break
            print(("Episode {}: reward = {}, steps = {}")\
                    .format(e + 1, total_reward, t))
