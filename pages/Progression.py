# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from PIL import Image

st.markdown("# Progression of Malarial Parasite")
st.markdown('To track the progression of malarial parasite we determined the ratio of parasite to the blood sample.')
image = Image.open('paths.jpeg')

st.image(image, caption='Stages of malaria progression')


st.sidebar.markdown("# Progression")