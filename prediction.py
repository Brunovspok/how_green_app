import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.preprocessing import StandardScaler

#creating dataframes


#energy production dataframe
df = pd.read_csv('data/full_data_2011-01-01_2022-11-26.csv', sep=',')
df = df.loc[[2]].drop(columns='Unnamed: 0').T
df.columns = ['energie']
df['The_date'] = df['energie']

#precipitation dataframe

df_rain = pd.read_csv('data/data_precipitation+2weeks.csv',sep=';')

#temperature dataframe

df_temp = pd.read_csv('data/temperature_data+2weeks.csv', sep=';')

#convert date to date_time series
dict_month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
for i in range(len(df)):
    day = "".join(re.findall("\d", df.index[i].split(" ", 1)[0]))
    month = dict_month[df.index[i].split(" ")[1]]
    year = "".join(re.findall("\d", df.index[i].split(" ", 1)[1]))
    the_date = f'{day}-{month}-{year}'
    df['The_date'].iloc[i] = the_date
df['The_date'] = pd.to_datetime(df['The_date'])


#mean values for each data frame

df_rain['mean'] = df_rain.mean(axis=1)
df_temp['mean'] = df_temp.mean(axis=1)


df['rain'] = df_rain['mean']
df['temp'] = df_temp['mean']

df['energie_ma'] = df['energie'].rolling(14).mean()
df['rain_ma'] = df['rain'].rolling(14).mean()
df['temp_ma'] = df['temp'].rolling(14).mean()


#Scaling data

df_to_scale = ['energie_ma','rain_ma', 'temp_ma']
scaler = StandardScaler()


for column in df_to_scale:
            df[column] = scaler.fit_transform(pd.DataFrame(df[column],columns=[column]))

X=[]
x=[]
y = []
z = []
Z = []
for i in range(14,1300):
    x = []
    x.append(df.loc[i, 'temp_ma'])
    x.append(df.loc[i, 'rain_ma'])
    x.append(df.loc[i, 'energie_ma'])
    X.append(x)

for i in range(1300,1500):
    y.append(df.loc[i, 'energie_ma'])

X_train = np.array(X).astype(np.float32)
y_train = np.array(y).astype(np.float32)

for i in range(1500,len(df)):
    z = []
    z.append(df.loc[i, 'temp_ma'])
    z.append(df.loc[i, 'rain_ma'])
    z.append(df.loc[i, 'energie_ma'])
    Z.append(x)

X_test = np.array([Z]).astype(np.float32)
