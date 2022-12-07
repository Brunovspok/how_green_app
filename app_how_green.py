import streamlit    as st
import requests
from streamlit_lottie import st_lottie
import datetime
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from datetime import date


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
        today=date.today()
        end_date = today + datetime.timedelta(days=5)
        prediction_day=st.date_input(label='Prediction for day:',min_value= today, max_value=end_date)
        prediction_element=st.selectbox(label='Element to predict', options=('Hidroelectric Energy Production','Temperature','Precepitation'))
        st.form_submit_button('Make prediction')

    with column2:
        #graphic
            fig = plt.figure(figsize=(15,8))
            plt.plot([1, 2, 3, 4, 5])
            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, height=800)
    params=dict(
        prediction_day=prediction_day,
        prediction_element=prediction_element)
    #green_app_url='API URL'
    #response=requests.get(green_app_url, params=params)
