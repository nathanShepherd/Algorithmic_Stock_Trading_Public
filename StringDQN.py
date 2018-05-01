# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

'''
    Note: String DQN is extremely sensitive to internal parameters
          Improve congvergance by solving subproblems with reward function
                                  picking a good set of range values for bins

'''

observe_training = False
EPSILON_MIN = 0.1
NUM_BINS = 8#must be even#
ALPHA = np.tanh
GAMMA = 0.9

EPOCHS = 10000

def max_dict(d):
    max_val = float('-inf')
    max_key = ""
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

class DQN:
    def __init__(self, obs_space, num_actions, observation, num_bins, bin_ranges=None):

        self.obs_space = obs_space
        self.num_actions = num_actions
        
        self.bin_ranges = bin_ranges
        self.bins = self.get_bins(num_bins)
        self.init_Q_matrix(observation)

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        self.Q = {}
        self.find(''.join(str(int(elem)) for elem in self.digitize(obs)))

    def find(self, state_string):
        try:
            self.Q[state_string]
        except KeyError as e:
            self.Q[state_string] = {}
            for action in range(self.num_actions):
                self.Q[state_string][action] = 0

    def get_action(self, state, use_policy=True):        
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.find(string_state)
        if use_policy:
            return max_dict( self.Q[string_state] )[0]
        else:
            return random.randint(0, self.num_actions - 1)

    def update_policy(self, state, state_next, action, reward):
        state_value = self.evaluate_utility(state)
        
        action = self.get_action(state)
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA(reward + GAMMA * reward_next - state_value)


        state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.Q[state][action] = state_value

    def evaluate_utility(self, state):
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.find(string_state)
        return max_dict( self.Q[string_state] )[1]


    def get_bins(self, num_bins):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(len(self.bin_ranges) == num_bins)
        ranges = [rng for rng in self.bin_ranges]

        bins = []
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = -ranges[i], ranges[i]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)
        return bins
            
    def digitize(self, arr):
        # distrubute each elem in state to the index of the closest bin
        state = np.zeros(len(self.bins))
        for i in range(len(self.bins)):
            state[i] = np.digitize(arr[i], self.bins[i])
        return state

    def get_state_stats(self):
        for i in range(len(self.Q)):
            print("\nElem:",i,end=" ")
            keys = [key for key in self.Q[i].keys()]
            print("Range: [%s, %s]" % (min(keys), max(keys)),
                  "STDDEV:", round(np.std(keys), 3), "Count:" , len(keys))


    def train(self, epochs, viz=False, agent=False):
        rewards = [0]; avg_rwd = 0
        EPSILON_MIN = 0
        dr_dt = 0#reward derivitive with respect to time
        
        for ep in range(1, epochs):
            #epsilon = max(EPSILON_MIN, np.tanh(-ep/(min(epochs, 2000)/2))+ 1)
            epsilon = 0.01       
            ep_reward = self.play_episode(epsilon, viz)
            if ep % 1 == 0:
                avg_rwd = round(np.mean(rewards),3)
                dr_dt = round(abs(dr_dt) - abs(avg_rwd), 2)
                print("Ep: {} | {}".format(ep, epochs),
                      "%:", round(ep*100/epochs, 2),
                      "Eps:", round(epsilon, 2),
                      "Avg rwd:", round(avg_rwd , 2),
                      "Ep rwd:", int(ep_reward),
                      "dr_dt:", dr_dt)

            if ep_reward < -1000: ep_reward = 0
            rewards.append(ep_reward)
            dr_dt = round(avg_rwd,2)

        return rewards

    def play_episode(self, epsilon=0.2, viz=False):
        state = self.env.reset()
        total_reward = 0
        terminal = False

        while not terminal:
            #if viz: env.render()
            #if num_frames > 300: epsilon = 0.1

            if random.random() < epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                action = self.get_action(state)
            
            state_next, reward, terminal, info = self.env.step(action)

            total_reward += reward
        
            if terminal:
                pass#shape reward
            
            action_next = self.get_action(state_next)
        
            self.update_policy(state, state_next, action, reward)
              
            state = state_next

        return total_reward



def observe(agent, N=15):
    [play_episode(agent, EPSILON_MIN, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def play_random(viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward
        
    return total_reward

def save_agent(A):
    with open('AlgoTrading_StringDQN.pkl', 'wb') as writer:
        pickle.dump(A, writer, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_agent(filename):
    with open(filename, 'rb') as reader:
        return pickle.load(reader)





if __name__ == "__main__":
    episode_rewards, num_frames, Agent = train(obs_space, act_space=action_space,
                                      epochs = EPOCHS, obs = observe_training)
    print("Completed Training")
    random_rwds = []
    for ep in range(EPOCHS):
        pass# The upper bound on random LunarLander is 0
        #random_rwds.append(play_random())

    plt.title("Average Reward with Q-Learning By Episode (LunarLander)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()

    recent_agent = 'Agent_LunarLander_strDQN.pkl'





















