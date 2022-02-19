import streamlit as st
import pandas as pd
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.datasets import fetch_openml
data_diabetes = fetch_openml(data_id=37)

# This is the whole sidebar menu:

st.sidebar.image('PicAndy2.PNG')
st.sidebar.subheader("Profile Summary :")
st.sidebar.write("15+ years of experience including 5 years in strategy consulting "
                 "and 5 years in senior management role")
st.sidebar.write("Management of over 50 people")
st.sidebar.write("Le Wagon DataScience / AI bootcamp")
st.sidebar.write("INSEAD Strategy certification")
st.sidebar.write("Expand the various dropboxes on the right to learn more "
                 "about my capabilities or to play around with simple ML models",bold=True)



# Introduction text

st.title('Andreas Jakobsson ')
st.header('Programming and Data Science')
st.write("""
             On this page you find information about my technical skills.
             This page should be complemented with my official CV or with my Linkedin profile.
             I have also added playground ML project for testing Streamlit capabilities.
         """)


# Building DataFrame and chart for DS and Algos expander below
df = pd.DataFrame({
    '# of Puzzles': [12,9,7,5,4,3,3,3,2],
    'DataStructures and Algortithms': ['01 Graph Theory', '02 Pathfinding', '03 BFS DFS', '04 Greedy Algorithms','05 Memoization',
                     '06 Minimax','07 Binary Search Tree','08 Simulation','09 Dynamic Programming'],

})

chart = (
    alt.Chart(df, title="Type of DataStructure or Algorithm used to solve Puzzle")
    .mark_bar()
    .encode(
        alt.X("DataStructures and Algortithms"),
        alt.Y("# of Puzzles"),
        #alt.Color("DS and Algos"),
        alt.Tooltip(["DataStructures and Algortithms", "# of Puzzles"]),
    )
    .interactive()
)



with st.expander("Data Structures and Algorithms", expanded=False):

    st.write("I have solved over 400 programming puzzles online on Codewars, Hackerrank and CodinGame")
    st.write(" - 150+ easy C++ puzzles on Codewars")
    st.write(" - 200+ easy Python puzzles on Codewars, Hackerrank and CodinGame")
    st.write(" - 70+ medium/advanced Python puzzles mainly on Codingame")
    st.write(" ")
    st.write("""Below a chart of advanced techniques or algorithms I have used to solve medium/advanced CodinGame puzzles. 
                Each puzzle takes between 30 minutes and 10 hours to solve. Basic techniques are omitted from the chart.
                """)
    st.altair_chart(chart,use_container_width=True)


with st.expander("AI Bot Programming", expanded=False):
    st.write("Developed 8 AI bots at Codingame and currently ranked top 0.5% on CodinGame. My bots typically achieve gold or legendary status")
    st.write("Type of bots developed:")
    st.write(" - Neural Network (trained in TensorFlow and built my own Feedforward network for efficient implementation")
    st.write(" - Rule based bots")
    st.write(" ")
    st.write("My bots are built in conjunction with various search algorithms (Beam Search, MCTS, Pathfinding)")
    st.write(" ")
    st.write("""Below some links to selected gameplays. Photosynthesis and PacMan Rock Paper Scissors
                are complex games with lots of edge cases while Line Racer and Great Escape are easier
                to build. Gamedetails and descriptions on codingame.com
            """)
    Photosyntlink = '[PhotoSynthesis GamePlay](https://www.codingame.com/replay/609318844)'
    st.markdown(Photosyntlink, unsafe_allow_html=True)
    PacManlink = '[PacMan Rock Paper Scissors GamePlay](https://www.codingame.com/replay/609318413)'
    st.markdown(PacManlink, unsafe_allow_html=True)
    lineRacerlink = '[Line Racer GamePlay](https://www.codingame.com/replay/609317576)'
    st.markdown(lineRacerlink, unsafe_allow_html=True)
    GreatEscapelink = '[Great Escape GamePlay](https://www.codingame.com/replay/609318686)'
    st.markdown(GreatEscapelink, unsafe_allow_html=True)

with st.expander("Machine Learning Logistics Regression Playground", expanded=False):
    st.write(" Overall goal is to classify if patient has diabetes")

    def diabetes_positive(row_input):
        if row_input == "tested_positive":
            return 1
        return 0

    y = data_diabetes['target'].apply(diabetes_positive)
    X = data_diabetes['data']

    st.write(" This is how the raw data looks like (source: openml(data_id=37), target column has been removed) :")

    st.table(X.head(2))

    cv_folds = st.slider("Select number of cross validation folds", 2, 20, 5, 1)
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(solver='lbfgs', random_state=42))
    ])
    final_score = cross_validate(pipe, X, y, cv=cv_folds, scoring=['accuracy', 'recall', 'precision', 'f1'])

    st.write(" As you modify the cross validation folds, you can see how the outcome is adjusted below:")

    st.write("Accuracy score : ", round(final_score['test_accuracy'].mean(),2), " | Standard dev : ",round(final_score['test_accuracy'].std(),2))
    st.write("Precision score : ", round(final_score['test_precision'].mean(),2), " | Standard dev : ",round(final_score['test_precision'].std(),2))
    st.write("Recall score : ", round(final_score['test_recall'].mean(),2), " | Standard dev : ",round(final_score['test_recall'].std(),2))
    st.write("F1 score : ", round(final_score['test_f1'].mean(),2), " | Standard dev : ",round(final_score['test_f1'].std(),2))
    st.write(" ")

    st.write("Model above is built as a very simple SKLearn pipeline with MinimaxScaler and Logisticregression")
    st.write("I have also put a random state for response consistency ")

with st.expander("Machine Learning NLP", expanded=False):
    st.write("""
             Details about the bots created
         """)

with st.expander("Programming languages", expanded=False):
    st.write("""
             Details about the bots created
         """)
with st.expander("Certifications", expanded=False):
    st.write("""
             Details about the bots created
         """)