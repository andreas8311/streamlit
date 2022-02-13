import streamlit as st
import pandas as pd
import plotly.graph_objects as go



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
    'first column': [12, 9, 3, 4, 5],
    'columns': ['Graph Theory', 'Path Finding', 'test3', 'test4', 'test5']
})


with st.expander("Data Structures and Algorithms", expanded=False):

    fig = go.Figure(go.bar(df, x='columns', y='first column'))

    # Plot!
    st.plotly_chart(fig, use_container_width=True)
    st.write("Here we write some useless stiff")
with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)