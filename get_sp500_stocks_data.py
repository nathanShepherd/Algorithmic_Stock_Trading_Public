# Get SP500 List from Wikipedia, Get stock data from Yahoo
# From tutorial by Sentdex @ https://bit.ly/2CyGCX5
# Written by Nathan Shepherd
import os
import pickle
import requests
import bs4 as bs

import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import preprocessing

#import pandas_datareader.data as web
import yfinance as yf


style.use('ggplot')

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.split('\n')[0])
    with open('sp500tickers.pkl', 'wb') as f:
        pickle.dump(tickers, f)
    return tickers
#save_sp500_tickers()

def load_sp500_tickers():
    t = []
    with open('sp500tickers.pkl', 'rb') as f:
        t = pickle.load(f)
    return t
    
def get_data_from_yahoo(N = 50):#max(N)=505
    if not os.path.exists('sp500stock_dfs'):
        os.makedirs('sp500stock_dfs')

    start = dt.datetime(2015, 12, 1)
    end = dt.datetime(2020, 12, 2)

    tickers = load_sp500_tickers()
    for i, ticker in enumerate(tickers[:N]):
        if i % int(N/10) == 0: print(i*100/N,'% complete')
        if not os.path.exists('sp500stock_dfs/%s.csv'%(ticker)):
            print('Fetching price data for %s'%(ticker))
            try:
                #df = web.get_data_yahoo(ticker, start, end)['prices']
                tickerData = yf.Ticker(ticker)
                df = tickerData.history(period='1d', start=start, end=end)
                df['Close'].to_csv('sp500stock_dfs/%s.csv'%(ticker))
            except KeyError as e:
                print('\t KeyError while fetching:', e)
            
        else:
            print('Already have %s' % (ticker))

#get_data_from_yahoo(505)

def compile_data():
    tickers = load_sp500_tickers()

    main_df = pd.DataFrame()

    for i, tick in enumerate(tickers):
        if os.path.exists('sp500stock_dfs/%s.csv'%(tick)):
            df = pd.read_csv('sp500stock_dfs/%s.csv'%(tick))
            df.set_index('Date', inplace=True)
            df.rename(columns = {'Close': tick}, inplace=True)

            if main_df.empty: main_df = df
            else: main_df = main_df.join(df, how='outer')

            if i % 10 == 0: print(i*100/len(tickers))
        else:print('Data for %s not found' % (tick))
        
    print(main_df.tail())
    main_df.to_csv('sp500_joined_Closed_prices.csv')

#compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_Closed_prices.csv')

    df_corr = df.corr()#correlates company price data
    df_corr.to_csv('sp500corr.csv')

    data = df_corr.values#returns np.array of rows and columns
    data.sort(axis=0)
    data.sort(axis=1)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    plt.savefig("correlations.png", dpi = (300))
    #plt.show()

#visualize_data()

def save_summary_graph():
    np.random.seed(406)
    df = pd.read_csv('sp500_joined_Closed_prices.csv')

    dates = df.iloc[:, 0]
    values = df.iloc[:,1:].interpolate(axis="columns").values
    

    #dates, values = values[]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(values)
    x_agg_mean = np.mean(x_scaled, axis=1)
    '''
    row_mode_arr = []
    for i in range(len(x_scaled[:,0])):
        mode_vals, counts = np.unique(x_scaled[i], return_counts=True)
        index = np.argmax(counts)
        row_mode_arr.append(mode_vals[index])

    row_mode_arr = np.array(row_mode_arr)
    '''
    randnorm_prices = np.random.normal(0.005,.1, size=x_scaled.shape)
    norm_agg_mean = np.mean(randnorm_prices, axis=1)

    randbeta = np.random.poisson(.006, size=x_scaled.shape)
    beta_agg_mean = np.mean(randbeta, axis=1)

    compile_df = pd.DataFrame({"mean_sp": x_agg_mean,
                               #"mode": row_mode_arr,
                               "rand_norm": norm_agg_mean,
                               "rand_poisson": beta_agg_mean},
                               index=dates)

    compile_df["rand_norm"] = compile_df["rand_norm"].rolling(100).sum()
    compile_df["rand_poisson"] = compile_df["rand_poisson"].rolling(100).sum()
    #compile_df.iloc[:-400].plot()
    compile_df.plot()
    plt.title("Random Walk Dist. Comparison")
    plt.tight_layout()
    plt.savefig("normalized_rowise_comparison.png", dpi = (300))

    print(compile_df.describe())
    

save_summary_graph()
















    
        
    
