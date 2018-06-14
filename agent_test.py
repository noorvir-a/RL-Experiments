import gym
import numpy as np

env_t = gym.make('CartPole-v0')
env_t._max_episode_steps = 301
MODEL_FOLDER = './models/'
GAMMA = 0.99

def test(X, graph, sess, num_test_runs, render=False):
    """
    Test agent's performance.
    :param env:
    :param X:
    :param graph:
    :param sess:
    :param num_test_runs:
    :param render:
    :return:
    """

    # List of episode lengths for each episode (number of time-steps)
    ep_length = []
    c_reward = []

    for run in range(num_test_runs):

        state = env_t.reset()
        ep_c_reward = 0
        reward = 0

        for t in range(300):

            if render:
                env_t.render()

            actions = sess.run(graph, feed_dict={X: [state]})
            action = np.argmax(actions)
            state, _, is_done, _ = env_t.step(action)
            ep_c_reward += (GAMMA ** t) * reward

            if is_done or t == 299:
                reward = -1
                ep_length.append(t+1)
                ep_c_reward += (GAMMA ** t) * reward
                c_reward.append(ep_c_reward)
                break

    return np.mean(c_reward), np.std(c_reward), np.mean(ep_length), np.std(ep_length)
