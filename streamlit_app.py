import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="My Interactive CV and AI portfolio",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )



st.write("Here's our first attempt at using data to create a table:")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
})


add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

with st.expander("Click here if you want to  know more", expanded=False):
    st.line_chart(data=df)
    st.write("Here we write some useless stiff")
with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)