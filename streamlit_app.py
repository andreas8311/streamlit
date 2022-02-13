import streamlit as st
import pandas as pd
import altair as alt


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
    'valuese': [9,8],
    'algos': ['Pathfinding', 'Graph Theory'],

})

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        alt.X("algos"),
        alt.Y("valuese"),

    )
    .interactive()
)



with st.expander("Data Structures and Algorithms", expanded=False):

    st.altair_chart(chart)
    st.write("Here we write some useless stiff")

with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)