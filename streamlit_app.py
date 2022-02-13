import streamlit as st
import pandas as pd


st.write("Here's our first attempt at using data to create a table:")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
})

# Random text

st.title('My title')
st.header('My header')
st.subheader('My sub')

st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *




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