# Get stock price data for a single company
# Tutorial by Siraj Raval --> https://youtu.be/SSu00IRRraY
# DATA FROM https://www.quandl.com/data/HKEX-Hong-Kong-Exchange/usage/quickstart/python
# Developed by Nathan Shepherd

import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import style
import matplotlib.pyplot as plt
#import pandas_datareader.data as web
import quandl
style.use('ggplot')

API_KEY = 'osvC1Uk4U-D7k9TQL3-z'
quandl.ApiConfig.api_key = API_KEY

data = quandl.get("EIA/PET_RWTC_D")
india_stocks = quandl.get('NSE/DCMFINSERV')
#print(data)

plt.plot(np.array(data['Value']))
plt.show()
