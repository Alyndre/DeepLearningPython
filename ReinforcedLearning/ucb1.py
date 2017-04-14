import numpy as np
import matplotlib.pyplot as plt
# plot vol dir dibuixar en un grafic

class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.random() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1-1.0/self.N) * self.mean + 1.0/self.N*x


def ucb(mean, n, nj):
    return mean + np.sqrt(2 * np.log(n) / (nj + 10e-3))


def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in xrange(N):
        # optimistic initial values
        j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print b.mean

    return cumulative_average

if __name__ == '__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 100000)

    plt.plot(c_1, label='eps = 0.1')
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(c_1, label='eps = 0.1')
    plt.legend()
    plt.show()