import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

matplotlib.use('Agg')

class Bandit:
    def __init__(self, arm, epsilon, initial):
        self.k = arm
        self.epsilon = epsilon
        self.initial = initial

        self.reset()

    def reset(self):
        """reset variables after every run
        """
        self.estimates =  np.zeros(self.k) + self.initial
        self.rewards = np.random.randn(self.k)

        self.action_count = np.zeros(self.k)

        self.best_action = 0
        self.average_reward = 0
        self.step = 0

    def act(self):
        """Choose action
            
            Returns:
                action(int)
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        else:
            return np.random.choice(np.where(self.estimates == np.max(self.estimates))[0])

    def is_best_action(self, action, rewards):
        """Judge if action has the best reward

            Args:
                action(int)
                rewards(array)
            Returns:
                True/False(bool)
        """
        return action == np.argmax(rewards)

    def update_estimate(self, action, reward):
        """Update estimatation of chosen action

            Args:
                action(int)
                reward(int)
        """
        self.action_count[action] += 1
        current_estimation = self.estimates[action]
        self.estimates[action] = current_estimation + ((reward - current_estimation) / self.action_count[action])

    def execute(self):
        """Execution of every step
        """
        self.step += 1

        action = self.act()
        # reward = self.rewards[action]
        reward = np.random.randn() + self.rewards[action]
        self.average_reward += (reward - self.average_reward) / self.step

        if self.is_best_action(action, self.rewards):
            self.best_action = 1
        else:
            self.best_action = 0

        self.update_estimate(action, reward)

def simulate(epsilons, run, step):
    """simulate tested arms
    """
    arm = 10
    initial = np.zeros(arm)

    best_action = np.zeros((len(epsilons), run, step))
    average_reward = np.zeros((len(epsilons), run, step))

    bandits = [Bandit(arm, eps, initial) for eps in epsilons]
    for i, bandit in enumerate(bandits):
        for r in trange(run):
            bandit.reset()
            for t in range(step):
                bandit.execute()
                best_action[i, r, t] = bandit.best_action
                average_reward[i, r, t] = bandit.average_reward

    best_action_each_run = best_action.mean(axis=1)
    average_reward_each_run = average_reward.mean(axis=1)

    return best_action_each_run, average_reward_each_run

def plot():
    epsilons = [0, 0.1, 0.01]
    run = 2000
    step = 1000

    best_choice, average_reward = simulate(epsilons, run, step)

    plt.subplot(2, 1, 1)
    for eps, reward in zip(epsilons, average_reward):
        plt.plot(reward, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, count in zip(epsilons, best_choice):
        plt.plot(count, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure.png')
    plt.show()

if __name__ == '__main__':
    plot()
