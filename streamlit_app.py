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



df = pd.DataFrame({
    '# of Puzzles': [12,9,7,5,4,3,3,3,2],
    'DS and Algos': ['01 Graph Theory', '02 Pathfinding', '03 BFS DFS', '04 Greedy Algorithms','05 Memoization',
                     '06 Minimax','07 Binary Search Tree','08 Simulation','09 Dynamic Programming'],

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

    st.write("I have solved over 400 programming puzzles online on Codewars, Hackerrank and CodinGame")
    st.write(" - 150+ easy C++ puzzles on Codewars")
    st.write(" - 200+ easy Python puzzles on Codewars, Hackerrank and CodinGame")
    st.write(" - 70+ medium/advanced Python puzzles on mainly on Codingame")
    st.write(" ")
    st.write("""Below a chart of advanced techniques or algorithms I have used to solve medium/advanced CodinGame puzzles. 
                Each puzzle takes between 30 minutes and 10 hours to solve. Basic techniques are omitted from the chart.
                """)
    st.altair_chart(chart,use_container_width=True)


with st.expander("AI Bot Programming", expanded=False):
    st.write("""
             Details about the bots created
         """)

with st.expander("Data Prep-Processing", expanded=False):
    st.write("""
             Details about the bots created
         """)
with st.expander("Machine Learning Linear Regression", expanded=False):
    st.write("""
             Details about the bots created
         """)
with st.expander("Machine Learning Logistics Regression", expanded=False):
    st.write("""
             Details about the bots created
         """)
with st.expander("Machine Learning NLP", expanded=False):
    st.write("""
             Details about the bots created
         """)