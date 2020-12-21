import numpy as np
import pandas as pd


np.random.seed(406)

def random_split():
    df = pd.read_csv('joined_df_stats/sp500_joined_Closed_prices.csv')

    col_idx = np.random.randint(1,500, size=10)

    sample = df.iloc[:, col_idx]

    lenX = np.floor(sample.shape[0] * 0.8)
    lenY = sample.shape[0] - lenX
    trainx, trainy = sample[:lenX], sample[:-lenY]
    
    import pdb; pdb.set_trace()

random_split()