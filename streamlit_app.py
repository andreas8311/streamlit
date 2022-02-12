import streamlit as st
import pandas as pd

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4, 5],
    'second column': [10, 20, 30, 40, 50]
}))


add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

with st.expander("Click here if you want to  know more", expanded=False):
    st.write("""
             And here you have some useless stuff.
         """)
    with st.expander("second stuff", expanded=False):
        st.write("""
                 and even more details
             """)