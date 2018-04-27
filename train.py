# Train an Algorithm to trade stocks for Profit
# Developed by Nathan Shepherd

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import numpy as np
import pickle


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
    outs = outs.T# IS TRANSPOSE OKAY?
    return outs, tickers


if __name__ == "__main__":
    data, ticks = get_data()
    
