import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import numpy as np

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)



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

"""
with st.expander("Data Structures and Algorithms", expanded=False):

    fig = px.bar(df, x='columns', y='first column')

    # Plot!
    st.plotly_chart(fig, use_container_width=True)
    st.write("Here we write some useless stiff")
"""
with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)