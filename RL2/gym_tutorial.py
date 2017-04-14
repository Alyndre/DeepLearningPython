import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt


def get_action(state, weights):
    return 1 if state.dot(weights) > 0 else 0


def episode(env, weights, render=False):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        if render:
            env.render()
        t += 1
        action = get_action(observation, weights)
        observation, reward, done, info = env.step(action)
        if done:
            break

    return t


def random_search(env):
    episode_lengths = []
    best = 0
    weights = None

    for t in xrange(100):
        new_weights = np.random.random(4) * 2 - 1
        curr_episode_length = np.empty(100)
        for i in xrange(100):
            curr_episode_length[i] = episode(env, new_weights)
        avg_length = curr_episode_length.mean()
        episode_lengths.append(avg_length)
        if avg_length > best:
            weights = new_weights
            best = avg_length

    return episode_lengths, weights


if __name__ == '__main__':
    environment = gym.make('CartPole-v0')
    l, w = random_search(environment)
    plt.plot(l)
    plt.show()

    environment = wrappers.Monitor(environment, 'D:\RL\RandomCartPole')
    for x in xrange(100):
        episode(environment, w, render=True)
