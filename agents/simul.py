import numpy as np

from utils.memory import Transition


class BaseSimulator(object):

    def __init__(self, agent, env):

        self.agent = agent
        self.env = env


class DQNSimulator(BaseSimulator):
    """Simulator for training DQN."""

    def __init__(self, agent, env, replay):

        super(DQNSimulator, self).__init__(agent, env)
        self.replay = replay


    def train(self, num_steps, target_update_steps=200, exploration_steps=0, batch_size=32,
              process_func=None, before_replay_process_func=None, save_path=None):

        # convenient abbr
        agent = self.agent
        env = self.env
        replay = self.replay

        # initialize state
        current_reward = 0.0
        state = env.reset()
        episode = 1

        last_t = 0
        for t in range(num_steps):
            # stack current state with previous states to choose an action
            if before_replay_process_func:
                state = before_replay_process_func(state)
            s_stack = replay.get_recent_states(state)

            # pick an action
            if process_func:
                s_stack = process_func(s_stack)

            action = self.agent.select_action(s_stack) if exploration_steps == 0 \
                else agent.random_action()
            next_state, reward, done, _ = env.step(action)

            # clip reward range
            reward = np.sign(reward)
            current_reward += reward

            # store recent observation in replay memory
            if before_replay_process_func:
                processed_state = before_replay_process_func(state)
                replay.remember(processed_state, action, reward)
            else:
                replay.remember(state, action, reward)
            
            if exploration_steps > 0:
                exploration_steps -= 1
                if exploration_steps == 0:
                    print("Done exploring")
            else:
                # train network if there are enough samples
                # in replay memory
                batch = replay.sample(batch_size)
                if process_func:
                    new_state = tuple(map(process_func, batch.state))
                    new_next_state = tuple(map(process_func, batch.next_state))
                    batch = Transition(new_state, batch.action, new_next_state, batch.reward)
                agent.update(batch)

                # update the target network after fixed steps
                if t % target_update_steps == 0:
                    agent.update_target_network()

            # move onto next state
            state = next_state
            if done: 
                # remember this last state with dummy action and reward
                replay.remember(None, -1, -1)

                print(("Episode {} at t = {}/{}: reward = {}, eps = {:.3f}, steps = {}, "
                       "replay_size = {}").format(
                    episode, t, num_steps, current_reward, agent.eps_current,
                    t - last_t + 1, replay.size()))
                if save_path and exploration_steps == 0:
                    agent.save_model(save_path)

                # reset everything
                state = env.reset()
                current_reward = 0.0
                episode += 1
                last_t = t
