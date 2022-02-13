import streamlit as st
import pandas as pd


df = pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
})

# Random text

st.title('Andreas Jakobsson Programming and Data Science')
st.write("""
             On this page you find a selection of baseline projects and technical skills.
             This should be complemented with my official CV or with my Linkedin profile
         """)

st.header('My header')
st.subheader('My sub')







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