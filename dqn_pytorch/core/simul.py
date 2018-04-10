import numpy as np


def train(agent, env, num_steps=10000, target_update_steps=200, 
          exploration_steps=0, save_path=None):
    """Train an agent in the given environment."""

    # initialize state
    current_reward = 0.0
    state = env.reset()
    episode = 1

    last_t = 0
    for t in range(num_steps):
        # pick an action
        action = agent.select_action(state) if exploration_steps == 0 \
            else agent.random_action()
        next_state, reward, done, _ = env.step(action)
        if done: 
            next_state = None

        # clip reward range
        reward = np.sign(reward)
        current_reward += reward

        # store transition in replay memory
        agent.remember(state, action, next_state, reward)
        
        if exploration_steps > 0:
            exploration_steps -= 1
            if exploration_steps == 0:
                print("Done exploring")
        else:
            # train network if there are enough samples
            # in replay memory
            agent.update()

            # update the target network after fixed steps
            if t % target_update_steps == 0:
                agent.update_target_network()

        # move onto next state
        state = next_state
        if done: 
            print("Episode {} at t = {}/{}: reward = {}, eps = {:.3f}, steps = {}".format(
                episode, t, num_steps, current_reward, agent.eps_current,
                t - last_t + 1))
            if save_path and exploration_steps == 0:
                agent.save_model(save_path)
            # reset everything
            state = env.reset()
            current_reward = 0.0
            episode += 1
            last_t = t


def test(agent, env, num_episodes, verbose=0):
    """ Test a trained agent in a given environment."""

    total_reward = 0.0

    for e in range(num_episodes):
        state = env.reset()
        done = False
        current_reward = 0
        while not done:
            env.render()
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            current_reward += reward
            state = next_state
        total_reward += current_reward
        if verbose == 1:
            print("Episode {}/{}: reward = {}".format(e + 1, num_episodes, 
                                                      current_reward))

    print("Average reward = {}".format(total_reward / num_episodes))
