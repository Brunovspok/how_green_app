import streamlit    as st
import requests
from streamlit_lottie import st_lottie

#load assets


def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding= load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_UbkeyZPVH7.json')
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
        st.write("About 60% of the energy used in Portugal comes from renewable energies, half of which comes from hydroelectric energy.")
        st.write("This source of energy has been threatened due to the severe drought that has been increasing in the last few years, along all territory.")
        st.write("##")
    with right_column:
        st_lottie(lottie_coding, height=300, key="dam energy")
#--secon headliner
with st.container():
    st.subheader("The question we want to answer:")
    st.header('"Should we continue invest in hydroelectric energy?"')
