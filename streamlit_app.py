import streamlit as st
import pandas as pd



# Introduction text

st.title('Andreas Jakobsson ')
st.header('Programming and Data Science')
st.write("""
             On this page you find a selection of my baseline projects and technical skills.
             This page should be complemented with my official CV or with my Linkedin profile
         """)


add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

df = pd.DataFrame({
    'Graph Theory': [9],
    'Path finding': [12],

})


with st.expander("Data Structures and Algorithms", expanded=False):

    st.bar_chart(df)
    st.write("Here we write some useless stiff")

with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)