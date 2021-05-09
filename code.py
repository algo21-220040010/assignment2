# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:44:14 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pykalman import KalmanFilter
from matplotlib import pyplot as plt

import statsmodels.api as sm
import os
os.getcwd()

f=open(r'C:\Users\46362\Desktop\高频交易论文\计算基差收窄\2015-05-20.csv')
IF = pd.read_csv(f, index_col=False)
f=open(r'C:\Users\46362\Desktop\高频交易论文\计算基差收窄\SH2015-05-20.csv')
SH = pd.read_csv(f, index_col=False)
df=pd.merge(IF,SH,on='time')
data = df[['price', 'buy_vol', 'sale_vol']]
data['vol'] = data.buy_vol+data.sale_vol
data['isBuy'] = (data.buy_vol-data.sale_vol>0)
data['time'] = pd.to_datetime(df.time)

# Start from 9.30, ending 15.30
y,m,d = data.time.iloc[0].year, data.time.iloc[0].month, data.time.iloc[0].day
start_time = pd.to_datetime(str(y)+'-'+str(m)+'-'+str(d)+' 9:30:00')
end_time = pd.to_datetime(str(y)+'-'+str(m)+'-'+str(d)+' 15:00:00')
morning_end_time = pd.to_datetime(str(y)+'-'+str(m)+'-'+str(d)+' 11:30:00')
afternoon_start_time = pd.to_datetime(str(y)+'-'+str(m)+'-'+str(d)+' 13:00:00')

data = data[(data.time>=start_time) & (end_time>=data.time)]
data = data.set_index('time')

print(data.head())
print(data.tail())

fig, ax1 = plt.subplots()
ax1.plot(data.price.values)
ax1.set_ylim(3400, 3800)
ax1.set_ylabel('price')
ax2 = ax1.twinx()
ax2.plot(data.vol.values, 'r-')
ax2.set_ylim(0, 1000)
ax2.set_ylabel('volume')
plt.show()

def Get_HFTProxy_And_Resampled_Price(data, top_quantile=0.1, price_method='vwap'):
    sorted_vol = data.vol.sort_values(ascending=False)
    if data.vol.sum()>0:
        HFTProxy = sorted_vol.iloc[:int(top_quantile*len(data))].sum()/data.vol.sum()*\
                        (2*(data.buy_vol.sum()-data.sale_vol.sum()>0)-1)
    else:
        HFTProxy = 0
    try:
        if price_method=='close':
            resampled_price = data.price.iloc[-1]
        if price_method=='open':
            resampled_price = data.price.iloc[0]
        if price_method=='vwap':
            try:
                resampled_price = np.average(data.price, weights=data.vol)
            except:
                resampled_price = data.price.iloc[-1]
    except:
        resampled_price = np.nan
    return (HFTProxy, resampled_price)

hft_price = data.groupby(pd.Grouper(freq='10S')).apply(Get_HFTProxy_And_Resampled_Price)
hft_price = hft_price[(hft_price.index<=morning_end_time) | (hft_price.index>=afternoon_start_time)]

hft = np.array([x[0] for x in hft_price]).reshape(len(hft_price), 1)
logprice = 1e3*np.diff(np.array([np.log(x[1]) for x in hft_price])).reshape((len(hft_price)-1,1))

kf = KalmanFilter(initial_state_mean=logprice[0],em_vars=['transition_covariance', 'observation_covariance'])
kf.em(logprice, n_iter=5)

# Kalman filter for true price and noise
(true_price, true_price_covariances) = kf.filter(logprice)

# Get price noise, permenant price impact:
price_noise = logprice - true_price
perm_impact = np.diff(true_price.ravel()).reshape(len(true_price)-1, 1)

X = sm.add_constant(hft[1:], prepend=False)
noise_VS_hft_model = sm.OLS(price_noise, X)
noise_VS_hft_reg = noise_VS_hft_model.fit()
print(noise_VS_hft_reg.summary())

discovery_VS_hft_model = sm.OLS(perm_impact, X[1:])
discovery_VS_hft_reg = discovery_VS_hft_model.fit()
print(discovery_VS_hft_reg.summary())