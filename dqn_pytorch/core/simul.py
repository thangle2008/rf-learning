import time


def train(agent, env, num_steps=10000, target_update_steps=200, save_path=None):
    """Train an agent in the given environment."""

    # initialize state
    current_reward = 0.0
    state = env.reset()
    episode = 1

    losses = []
    for t in range(num_steps):
        # pick an action
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done: 
            next_state = None

        # clip reward range
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        current_reward += reward

        # store transition in replay memory
        agent.remember(state, action, next_state, reward)
        
        # optimize if there is enough in replay memory
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        # update the target network after fixed steps
        if t % target_update_steps == 0:
            agent.update_target_network()

        # move onto next state
        state = next_state
        if done: 
            if len(losses) != 0:
                average_loss = sum(losses) / len(losses)
            else:
                average_loss = -1.0
            print("Episode {} at t = {}/{}: reward = {}, eps = {:.3f}, avg_loss = {:.5f}".format(
                episode, t, num_steps, current_reward, agent.eps_current,
                average_loss))
            state = env.reset()
            current_reward = 0.0
            episode += 1
            losses = []
            
            if save_path:
                agent.save_model(save_path)


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
