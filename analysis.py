import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


np.random.seed(406)

def random_split():
    df = pd.read_csv('joined_df_stats/sp500_joined_Closed_prices.csv')

    col_idx = np.random.randint(1,500, size=10)

    sample = df.iloc[:, col_idx]

    lenX = np.floor(sample.shape[0] * 0.8)
    lenY = sample.shape[0] - lenX
    trainx, trainy = sample[:lenX], sample[:-lenY]
    
    import pdb; pdb.set_trace()

def calc_normal_returns():
    np.random.seed(406)
    df = pd.read_csv('joined_df_stats/sp500_joined_Closed_prices.csv')

    df.plot()
    plt.show()
    plt.savefig("joined_cl.png")

    dates = df.iloc[:, 0]
    values = df.iloc[:,1:].interpolate(axis="columns").values
    values_trim = values[:-1]
    #import pdb; pdb.set_trace()

    #dates, values = values[]
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(values)

    x_agg_mean = np.mean(x_scaled, axis=1)

    randnorm_prices = np.random.normal(0.005,.1, size=x_scaled.shape)
    norm_agg_mean = np.mean(randnorm_prices, axis=1)
    '''
    compile_df = pd.DataFrame({"mean_sp": x_agg_mean,
                               #"mode": row_mode_arr,
                               "rand_norm": norm_agg_mean,
                              },
                               index=dates)
                            

calc_normal_returns()