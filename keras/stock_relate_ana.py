import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts

wanke = '000002'
baoli = '600048'
beginDate = '2018-01-01'
endDate = '2019-04-04'
df_wanke=ts.get_hist_data(wanke,start=beginDate,end=endDate).sort_index(axis=0,ascending=True)
df_baoli=ts.get_hist_data(baoli,start=beginDate,end=endDate).sort_index(axis=0,ascending=True)
df=pd.concat([df_wanke['close'],df_baoli['close']],axis=1,keys=['wanke_close','baoli_close'])
df.ffill(axis=0,inplace=True)
df.to_csv('wanke_baoli.csv')
corr=df.corr(method='pearson',min_periods=1) #皮尔森相关性系数 
corr2 = df.corr(method='spearman')
corr3 = df.corr(method='kendall')
print(corr,corr2,corr3)
df.plot(figsize=(20,12))
plt.savefig('wanke_baoli.png')
plt.close()
wanke_max = np.max(df['wanke_close'])
wanke_min = np.min(df['wanke_close'])
wanke_mean = np.average(df['wanke_close'])
baoli_max = np.max(df['baoli_close'])
baoli_min = np.min(df['baoli_close'])
baoli_mean = np.average(df['baoli_close'])
df['wanke_one']=(df['wanke_close']-wanke_mean) / float(wanke_max - wanke_min) *100
df['baoli_one']=(df['baoli_close']-baoli_mean)/float(baoli_max-baoli_min)*100
df['wanke_one'].plot(figsize=(20,12))
df['baoli_one'].plot(figsize=(20,12))
plt.savefig('one.png')