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
    '# of Puzzles': [12,9,5,5,4,3,3],
    'DS and Algos': ['01 Graph Theory', '02 Pathfinding', '03 Greedy Algorithms','04 BFS','05 Memoization','06 Minimax','07 Binary Search Tree'],

})

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        alt.X("DS and Algos"),
        alt.Y("# of Puzzles"),
        #alt.Color("DS and Algos"),
        alt.Tooltip(["DS and Algos", "# of Puzzles"]),
    )
    .interactive()
)



with st.expander("Data Structures and Algorithms", expanded=False):

    st.write("""Below a chart of advanced techniques or algorithms I have used to solve CodinGame puzzles. 
                Each puzzle takes between 30 minutes and 10 hours to solve. Basic techniques are omitted from the chart.
                """)
    st.altair_chart(chart,use_container_width=True)
    st.write("Here we write some useless stiff")

with st.expander("second stuff", expanded=False):
    st.write("""
             and even more details
         """)