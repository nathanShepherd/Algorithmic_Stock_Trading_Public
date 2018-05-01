# Train an Algorithm to trade stocks for Profit
# Developed by Nathan Shepherd

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import numpy as np
import random
import pickle

#data has very low variability
def get_data():
    df = pd.read_csv('sp500_joined_Adj_Closed_prices.csv', index_col=0)
    df.fillna('null', inplace=True)
    tickers = df.columns.values.tolist()
    matrix = list(df.values)
    cleaned = []
    for col in range(len(matrix)):
        count = 0# avg=35, max=92
        for row in range(len(matrix[col])):
            if 'null' == matrix[col][row]:
                count += 1
        if count < 35:
            for i in range(len(matrix[col])):
                if matrix[col][i] == 'null':
                    matrix[col][i] = 0
                    
            cleaned.append(matrix[col])
        
    outs = np.array(cleaned)
    outs = outs.T# IS TRANSPOSE OKAY? Yes, but error when graphing
    return outs, tickers

def visualize(data):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data)#, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    #column_labels = data.columns
    #row_labels = data.index
    #ax1.set_xticklabels(column_labels)
    #ax1.set_yticklabels(row_labels)
    #plt.xticks(rotation=90)
    #heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

def mean_squared_err(f_x, g_x):
    total = 0; N = len(f_x); devi = []
    for i in range(N):
        total += (f_x[i] - g_x[i])**2
        devi.append((f_x[i] - g_x[i])**2)
    return total/N, max(devi)

def variance(y):
    mu = np.mean(y); N = len(y)
    vari = [(y[i] - mu)**2 for i in range(N)]
    avg_var = sum(vari)/N
    return avg_var

def evaluate(vect, other):
    out = []
    out.append(round(mean_squared_err(vect, other)[0], 3))
    out.append(round(variance(vect), 3))
    out.append(round(np.std(vect), 3))
    out.append(round(mean_squared_err(vect, other)[1], 3))
    return out

def get_training_data(data, deg=1, viz=False):
    cleaned = []#TODO: Pick samples with low range and high variance
    #data_pts = []
    #analyze variance to pick good samples of stock data
    N = len(data[0])
    concave_up = []
    quality_vect = [0, 0, 0, 0]
    
    for k, d in enumerate(data):
        x = [i for i in range(N)]
        coef = np.polyfit(x, d, deg)
        funct = np.poly1d(coef)
        for i, num in enumerate(evaluate(d, funct(x))):
            quality_vect[i] += num

        if coef[0] < 0 and viz and False:
            print(coef)
            print(k, end='\n\n')
            plt.plot(funct(x), 'purple', label='Fitted')
            plt.plot(d, 'black', label="Actual")
            plt.show()    

    quality_vect = [quality_vect[i]/len(data) for i in range(len(quality_vect))]
    for d in data:
        x = [i for i in range(N)]
        coef = np.polyfit(x, d, deg)
        funct = np.poly1d(coef)
        good = False
        for i, num in enumerate(evaluate(d, funct(x))):
            negative_slope = coef[0] < 0
            if quality_vect[i] < num or negative_slope:
                good = True
        if good:
            cleaned.append(d)
            if viz:
                plt.plot(funct(x), 'green', label='Fitted')
                plt.plot(d, 'black', label="Actual")
                plt.show()
    print("Accepted as above average", len(cleaned))
        
def get_subset(data):
    pts = [5, 7, 8, 9, 11, 12, 13, 15, 17, 18,#hand-picked
           20, 21, 27, 29, 30, 37, 44, 49, 50,
           58, 60, 67, 70, 75, 76 , 77, 89, 97, 101]
    cleaned = []
    for i, d in enumerate(data):
        if i in pts: cleaned.append(d)
    return cleaned


def split(data, ratio=0.8):
    N_x = int( len(data) * ratio)
    N_y = - int( len(data) * (1 - ratio))
    x = data[ :N_x ] ; y = data[ N_y: ]
    return x, y

class Env():
    def __init__(self, obs_sequence):
        self.time_step = 0
        self.data = obs_sequence
        self.end = len(obs_sequence)

        self.num_shares = 0
        self.total_funds = int(10000)

    def reset(self):
        self.time_step = 0
        self.num_shares = 0
        self.total_funds = int(10000)
        return data[0]

    def step(self, action):
        label = ['Buy', 'Sell', 'Hold'][action]
        #share value estimate could be improved?
        share_value = self.data[ self.time_step ][-1]
        starting_total = self.total_funds

        if 'Buy' == label:
            self.total_funds -= share_value
            if int(self.total_funds) <= 0:
                self.total_funds = 0
            else:
                self.num_shares += 1
            
        if 'Sell' == label:
            self.total_funds += share_value * self.num_shares
            self.num_shares = 0
        
        if 'Hold' == label: pass
        
        self.time_step += 1
        
        terminal = self.time_step == self.end - 1
        reward = self.total_funds - starting_total
        
        return self.data[self.time_step], reward, terminal, label

def convert_to_obs(sample, step_size= 5, poly_deg=3, binary=True):
    #Convert the raw share price into a more informative vector
    #print("Fitting polynomials and converting observation vector")
    x = [i for i in range(step_size)]
    outs = []
    for i in range(len(sample) - step_size):
        y = sample[i : i + step_size]
        coef = np.polyfit(x, y, poly_deg)
        funct = np.poly1d(coef)

        if binary: concavity = int(coef[0] <= 0)
        else: concavity = np.tanh(coef[0])
        #range_val = np.tanh(max(y) - min(y))
        #next_val = np.tanh(funct(step_size + 1)/100)
        #actual = np.tanh(y[-1]/100)

        obs_vect = [concavity]#, next_val, range_val, actual]
        outs.append(obs_vect)
        
    return outs

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def Plot(data, pred=None):
    #plot data and predictions
    #display average returns over all data (also, per data)
    print('Plotting episode replay')
    for i in range(len(data)):
        y = data[i]
        plt.plot(y, "black")
        if pred:
            color = ['green', 'red', 'blue', 'purple']
            #change color w.r.t. action
            plt.plot(0.5, color[pred[i]])

    plt.show()
    
def ML(data):
    #Use String DQN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--> learn a model-free policy that maximizes returns
    binary_data = [convert_to_obs(d) for d in data]
    train, test = split(binary_data, ratio = 0.8)
    from StringDQN import DQN
    obs_space = len(train[0][0]); num_actions = 3

    num_bins = 8; observation = train[0][0];
    bin_ranges = [1 for i in range(num_bins)]

    Agent = DQN(obs_space, num_actions, observation, num_bins, bin_ranges)

    rewards = []
    mean_rewards = 0
    for i, episode in enumerate(train):
        print("Episode %s of %s | Avg Returns %s" % (i, len(train), mean_rewards))
        Agent.env = Env(episode)
        ep_rewards = Agent.train( 2 )# num_epochs
        rewards.append( ep_rewards )
        mean_rewards += np.mean(ep_rewards)
        #plot_running_avg(ep_rewards)
        
    print("Total Returns:", sum([sum(ep) for ep in rewards]))
    #plt.show()

    #Use policy gradients to learn a model ~~~~~~~~~~
    #--> predict a policy that would maxamize returns

    
    #Use mathematical model to describe the state space ~~~
    #--> make predictions from extrapolating with the model
    print("Using mathematical model to make predictions")
    total_returns = 0
    tanh_data = []
    for i, d in enumerate(data):
        perc_done = i*100/(len(data))
        if int(perc_done) % 5 == 0: print(perc_done)
        tanh_data.append(convert_to_obs(d, binary=False))
        
    train, test = split(tanh_data, ratio = 0.8)
    for i, episode in enumerate(train):
        terminal = False
        total_reward = 0
        env = Env(episode)
        state = env.reset()

        actions = [2]
        while not terminal:
            if state[0] <= 0:
                #label = ['Buy', 'Sell', 'Hold'][action]
                if random.random() <= 0.99: action = 0
                else: action = 1
                state, reward, terminal, label = env.step(action)
            else:
                action = 2
                state, reward, terminal,_ = env.step(action)

            total_reward += reward
            actions.append(action)

##        y = data[i]
##        plt.plot(y, 'black')
##        color = ['green', 'red', 'blue', 'purple']
##        mu = np.mean(data)
##        for a in actions:
##            print(a)
##            plt.plot(mu, 'green')
##        plt.show()
        #Plot(data[i])#, actions)
        print("Episode: %s of %s | ep_rwd: %s" % (i, len(train), total_reward))
        total_returns += total_reward
        
    print("Amount Earned:", total_returns)
    

    
if __name__ == "__main__":
    print("Fetching stored sp500 Stock Data")
    data, ticks = get_data()
    #get_training_data(data)# for testing

    print("Searching for optimal training subset")
    data = get_subset(data)#Use subset of data
    
    #data = data[:100]
    '''
    Use simple trading strategy to make predictions
    pred = max(buy, hold, sell) '''
    print("Building and Evaluating Machine Learning Models")
    ML(data)

    Plot(data)
    
    
    
    
