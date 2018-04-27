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
import pandas_datareader.data as web


style.use('ggplot')

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open('sp500tickers.pkl', 'wb') as f:
        pickle.dump(tickers, f)
    return tickers

def load_sp500_tickers():
    t = []
    with open('sp500tickers.pkl', 'rb') as f:
        t = pickle.load(f)
    return t
    
def get_data_from_yahoo(N = 50):#max(N)=505
    if not os.path.exists('sp500stock_dfs'):
        os.makedirs('sp500stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2017, 12, 31)

    tickers = load_sp500_tickers()
    for i, ticker in enumerate(tickers[:N]):
        if i % int(N/10) == 0: print(i*100/N,'% complete')
        if not os.path.exists('sp500stock_dfs/%s.csv'%(ticker)):
            print('Fetching price data for %s'%(ticker))
            try:
                df = web.get_data_yahoo(ticker, start, end)['prices']
                df.to_csv('sp500stock_dfs/%s.csv'%(ticker))
            except KeyError as e:
                print('\t KeyError while fetching:', e)
            
        else:
            print('Already have %s' % (ticker))

def compile_data():
    tickers = load_sp500_tickers()

    main_df = pd.DataFrame()

    for i, tick in enumerate(tickers):
        if os.path.exists('sp500stock_dfs/%s.csv'%(tick)):
            df = pd.read_csv('sp500stock_dfs/%s.csv'%(tick))
            df.set_index('Date', inplace=True)
            df.rename(columns = {'Adj Close': tick}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty: main_df = df
            else: main_df = main_df.join(df, how='outer')

            if i % 10 == 0: print(i*100/len(tickers))
        else:print('Data for %s not found' % (tick))
        
    print(main_df.tail())
    main_df.to_csv('sp500_joined_Adj_Closed_prices.csv')

def visualize_data():
    df = pd.read_csv('sp500_joined_Adj_Closed_prices.csv')

    df_corr = df.corr()#correlates company price data
    df_corr.to_csv('sp500corr.csv')
    data = df_corr.values#returns np.array of rows and columns
    

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
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()



















    
        
    
