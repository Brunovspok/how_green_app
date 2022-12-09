import streamlit    as st
import requests
from streamlit_lottie import st_lottie
import datetime
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from datetime import date
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import StandardScaler
from pandas import to_datetime
from datetime import timedelta
    #load assets

#Theme set up on  "config.toml" -> HEX colour theme background - #eddbc3


def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding= load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_UbkeyZPVH7.json')
lottie_coding2=load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_iombyzfq.json')
# page title
st.set_page_config(page_title="How Green Can We Go?",page_icon=":deciduous_tree:",layout="wide")

#Header
with st.container():
    st.title(":seedling: How Green Can We Go? :seedling: ")
    st.header("Predicting Portugal's green energy production from dams, for a near future")
#--Introduction section
with st.container():
    st.write('---')
    st.write('##')
    left_column, right_column =st.columns(2)
    with left_column:
        st.write("In the last couple of decades the green energy has proven to be a promising alternative to fossil fuels,")
        st.write("although climate changes may have an impact of all that has been done until now.")
        st.write("About 60% of the energy used in Portugal comes from renewable energies, 40% of which comes from hydroelectric energy.")
        st.write("This source of energy has been threatened due to the severe drought that has been increasing in the last few years, along all territory.")
        st.write("##")
    with right_column:
        #st_lottie(lottie_coding2, height=300, key="sun")
        st_lottie(lottie_coding, height=300, key="dam energy")
st.write('---')
#-intro second part
with st.container():
    intro_column1,intro_column2, intro_column3=st.columns([1,2,1])
    with intro_column1:
        st.write("Since 2000 Portugal has been trough severe drought episodes, and every year this phenomenon becomes more and more extreme")
        st.write("As the drought itensifies, the southern part of the country is already dry, and these dams are already usting its capacity to local consumption only")
        st.write("Therefore, we only using data from the north region of the country(Douro) as the main source of hidroelectric production.")
    with intro_column2:
        img1_url='https://www.cruzeiros-douro.pt/content/uploads/maingallery/crops/596_banner_1568998997.jpg'
        st.image(image=img1_url,use_column_width='auto')
    with intro_column3:
        img2_url='https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/LocalRegiaoNorte.svg/1200px-LocalRegiaoNorte.svg.png'
        st.image(image=img2_url,width=260)
    st.write('---')
    st.write('##')

#--secon headliner
with st.container():
    st.header('Lets do some predictions! :four_leaf_clover:')

#st.slider(label='Prediction day:', min_value='2018', max_value='2032')


with st.form(key='params_for_api'):

    column1, column2= st.columns([1,3])
    with column1:
        #buttons/imputs
        min_val = datetime.date(2019, 1, 1)
        today=date.today()
        end_date = today + datetime.timedelta(days=48)
        prediction_day=st.date_input(label='Prediction for day:',min_value= min_val, max_value=end_date)
        prediction_element=st.selectbox(label='Element to predict', options=('Hidroelectric Energy Production','Temperature','Precepitation'))
        st.form_submit_button('Make prediction')




    with column2:
        ###Model

        #1676 = TODAY
        the_input = prediction_day #change
        today=date.today()
        date_difference = today - to_datetime(the_input).date()
        begin_date = 1676 - date_difference.days

        #downlaod data + prepocessing#
        df = pd.read_csv('data/full_data_2011-01-01_2022-11-26.csv')
        df_rain = pd.read_csv('data/data_precipitation.csv', sep=';')
        df_temp = pd.read_csv('data/temperature_data.csv', sep=';')
        df_futur_rain = pd.read_csv('data/future_rain.csv', sep=';')
        df_futur_temp = pd.read_csv('data/future_temp.csv', sep=';')


        df = df.loc[[2]].drop(columns='Unnamed: 0').T
        df.columns = ['energie']
        df['The_date'] = df['energie']
        dict_month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
        for i in range(len(df)):
            day = "".join(re.findall("\d", df.index[i].split(" ", 1)[0]))
            month = dict_month[df.index[i].split(" ")[1]]
            year = "".join(re.findall("\d", df.index[i].split(" ", 1)[1]))
            the_date = f'{day}-{month}-{year}'
            df['The_date'].iloc[i] = the_date
        df['The_date'] = pd.to_datetime(df['The_date'])
        df_temp.drop(columns=['Unnamed: 17'], inplace = True)
        df = df[2557:-18]
        df.reset_index(inplace = True)
        df_futur_rain['mean'] = df_futur_rain.mean(axis=1)
        df_futur_temp['mean'] = df_futur_temp.mean(axis=1)

        df_final = pd.DataFrame()
        df_final['temp'] = df_futur_temp['mean']
        df_final['rain'] = df_futur_rain['mean']
        df_final['energie'] = df["energie"]

        #smooth curve#
        df_final['energie_ma'] = df_final['energie'].rolling(14).mean()
        df_final['rain_ma'] = df_final['rain'].rolling(14).mean()
        df_final['temp_ma'] = df_final['temp'].rolling(14).mean()

        #data scaling#
        df_to_scale = ['energie_ma','rain_ma', 'temp_ma']
        scaler = StandardScaler()

        for column in df_to_scale:
                    df_final[column] = scaler.fit_transform(pd.DataFrame(df_final[column],columns=[column]))


        #data spliting(train-test-pred)#
        X=[]
        x=[]
        y = []

        for i in range(14,begin_date):
            x = []
            x.append(df_final.loc[i, 'temp_ma'])
            x.append(df_final.loc[i, 'rain_ma'])
            x.append(df_final.loc[i, 'energie_ma'])
            X.append(x)

        y = df_final['energie_ma'][begin_date:begin_date+48]

        X_train = np.array(X).astype(np.float32)
        y_train = np.array(y).astype(np.float32)

        dataset_test = timeseries_dataset_from_array(
            X_train,
            y_train,
            sequence_length=50,
            batch_size=32,
        )


        url = "https://how-green-cfddvd7twq-ew.a.run.app/predict/"
        #url = "http://127.0.0.1:8000/predict/"
        files = [("files", X_train),
         ("files", y_train)]

        response =requests.post(url,files=files).json()
        pred = np.array(response["pred"])

        #graphic
        df_final['pred'] = df_final['energie_ma']
        for i in range(48):
            df_final['pred'][begin_date+i] = pred[i]

        df_final['pred'] = scaler.inverse_transform(df_final[['pred']])
        df_final['energie_ma'] =scaler.inverse_transform(df_final[['energie_ma']])

        df_final['days'] = df_final.index
        for i in range(len(df_final)):
            df_final['days'].iloc[i] = datetime.datetime.today().date() - timedelta(int(df_final['days'].iloc[i]))
        df_final.set_index('days',inplace=True)

        fig = plt.figure(figsize=(15,8))
        plt.xlabel("Date")
        plt.ylabel("Energy Production (GW)")
        plt.title("Hidroelectric Production")

        if begin_date >= 1675:
            plt.plot(df_final['pred'][begin_date-100:begin_date],label='REAL');
            plt.plot(df_final['energie_ma'][begin_date-1:begin_date+48],ls='--',label='PREDICTION');
            plt.legend(loc='upper left', fontsize=8);
        else:
            plt.plot(df_final['pred'][begin_date-30:begin_date+48],ls='--',label='PREDICTION');
            plt.plot(df_final['energie_ma'][begin_date-30:begin_date+48],label='REAL');
            plt.legend(loc='upper left', fontsize=8);

        st.pyplot(fig)
