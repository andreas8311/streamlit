import streamlit as st
import pandas as pd




# Introduction text

st.title('Andreas Jakobsson Programming and Data Science')
st.write("""
             On this page you find a selection of my baseline projects and technical skills.
             This page should be complemented with my official CV or with my Linkedin profile
         """)


add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

df = pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
}, columns=['test1', 'test2', 'test3', 'test4', 'test5'])


with st.expander("Data Structures and Algorithms", expanded=False):
    st.bar_chart(data=df)
    st.write("Here we write some useless stiff")
with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)