
#Imports
import streamlit as st
import pandas as pd
import altair as alt # Chart module

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.datasets import fetch_openml
data_diabetes = fetch_openml(data_id=37) # diabetes dataset from openml

#########################
##########################
import requests
from datetime import date, timedelta, datetime
# import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import matplotlib.image as mpimg


st.title('MeteoStat')

def iteration_15min(start, finish):
    ## Generateur de (an, mois, jour, heure, minute)
     while finish > start:
        start = start + timedelta(minutes=15)
        yield (start.strftime("%Y"),
               start.strftime("%m"),
               start.strftime("%d"),
               start.strftime("%H"),
               start.strftime("%M")
               )

def open_save_data(url, date_save):
    ## Ouvre l'image pointee par url
    ## Enregistre l'image avec l'extention date_save

    print(url, date_save)

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    # img.save( f"./radar{date_save}.png")
    gray_image = mpimg.imread(img)
    print(gray_image.shape)
    return gray_image

def scrapping_images (start, finish) :
    """Scrape images radar en ligne toutes les 15 min
    entre deux dates donnees sous forme de datetime.datetime
    Sauvegarde les dates pour lesquelles la page n'existe pas.  """
    missing_times = []
    saved_images = []
    for (an, mois, jour, heure, minute) in iteration_15min(start, finish):
        ## url scrapping :
        url = (f"https://static.infoclimat.net/cartes/compo/{an}/{mois}/{jour}"
            f"/color_{jour}{heure}{minute}.jpg")
        date_save = f'{an}_{mois}_{jour}_{heure}{minute}'

        try :
            tmp = open_save_data(url, date_save)
            saved_images.append(tmp)


        except UnidentifiedImageError :
            print (date_save, ' --> Missing data')
            missing_times.append(date_save)

    ## Save missing data list :
    missing_data_name = f'missing_datetimes_{start.strftime("%Y")}\
        {start.strftime("%m")}{start.strftime("%d")}_to_{finish.strftime("%Y")}\
            {finish.strftime("%m")}{finish.strftime("%d")}'
    pd.DataFrame(missing_times).to_pickle(missing_data_name)
    print(missing_times)
    return saved_images

#def open_data(date_save):
    #print('Open '+date_save)
    #img = mpimg.imread(f"images/radar{date_save}.png")
    #return img

if st.button('Scrapping'):

    start = datetime(2022, 1, 20, 18,00)
    finish = datetime(2022, 1, 20, 20, 30)

    tmp = scrapping_images (start, finish)
    tmp = np.array(tmp)
    st.write(tmp.shape)




############################
############################



### Sidebar frame menu begins here:

st.sidebar.image('PicAndy2.PNG')
st.sidebar.subheader("Profile Summary :")
st.sidebar.write("15+ years of experience including 5 years in strategy consulting "
                 "and 5 years in senior management role")
st.sidebar.write("Management of over 50 people")
st.sidebar.write("Le Wagon DataScience / AI bootcamp")
st.sidebar.write("INSEAD Strategy certification")
st.sidebar.write("Expand the various dropboxes on the right to learn more "
                 "about my capabilities or to play around with a simple ML model",bold=True)
### Sidebar frame ends here


### Main frame

# Introduction text

st.title('Andreas Jakobsson ')
st.header('Programming and Data Science')
st.write("""
             On this page you find information about my technical skills.
             This page should be complemented with my official CV or with my Linkedin profile.
             I have also added playground ML project for testing Streamlit capabilities. The source code
             can be found on my Github by clicking "view app source" in the menu.
         """)


### Building DataFrame and chart for DS and Algos expander below
df = pd.DataFrame({
    '# of Puzzles': [12,9,7,5,4,3,3,3,2],
    'DataStructures and Algorithms': ['01 Graph Theory', '02 Pathfinding', '03 BFS DFS', '04 Greedy Algor.','05 Memoization',
                     '06 Minimax','07 Search Tree','08 Simulation','09 Dynamic Progr.'],

})

chart = (
    alt.Chart(df, title="Type of DataStructure or Algorithm used to solve Puzzle")
    .mark_bar()
    .encode(
        alt.X("DataStructures and Algorithms"),
        alt.Y("# of Puzzles", scale=alt.Scale(domain=(0,13))),
        alt.Tooltip(["DataStructures and Algorithms", "# of Puzzles"]),
    ).configure_axis(
    grid=False
)
    .interactive()
)
### End of DataFrame and Chart builder


### Expander sections begins here. Each expander is self-explanatory
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


with st.expander("AI Bot Programming and Reinforcement Learning", expanded=False):
    st.write("Developed 8 AI bots and currently ranked top 0.5% on CodinGame. My bots typically achieve gold or legendary status")
    st.write("Type of bots developed:")
    st.write(" - Neural Network (trained in TensorFlow Keras and built my own Feedforward network for efficient implementation)")
    st.write(" - Rule based bots")
    st.write(" ")
    st.write("My bots are built in conjunction with various search algorithms (Beam Search, MCTS, Pathfinding etc)")
    st.write(" ")
    st.write("""Below some links to selected gameplays. Photosynthesis and PacMan Rock Paper Scissors
                are complex games with lots of edge cases while Line Racer and Great Escape are easier
                to build. Game details and descriptions on codingame.com
            """)
    Photosyntlink = '[PhotoSynthesis GamePlay](https://www.codingame.com/replay/609318844)'
    st.markdown(Photosyntlink, unsafe_allow_html=True)
    PacManlink = '[PacMan Rock Paper Scissors GamePlay](https://www.codingame.com/replay/609318413)'
    st.markdown(PacManlink, unsafe_allow_html=True)
    lineRacerlink = '[Line Racer GamePlay](https://www.codingame.com/replay/609317576)'
    st.markdown(lineRacerlink, unsafe_allow_html=True)
    GreatEscapelink = '[Great Escape GamePlay](https://www.codingame.com/replay/609318686)'
    st.markdown(GreatEscapelink, unsafe_allow_html=True)

with st.expander("Data Science and Machine Learning", expanded=False):
    st.write("""
             Below, you can find a selection of packages I have used and models I have built:
         """)
    st.write(" ")
    st.write(" - Pandas, Scipy, Numpy, SKLearn, Jupyter Notebook, Matplotlib, Seaborn, Plotly")
    st.write(" - Supervised : Linear Regression, Logistic Regression, GridSearch, RandomSearch,Pipeline")
    st.write(" - Unsupervised : KMeans, PCA")
    st.write(" - TimeSeries : Arima, Sarima")

with st.expander("Deep Learning", expanded=False):
    st.write("""
             I have built my own Neural Network from scratch and in several occasions trained a model in tensorflow
             and thereafter imported the weights to my own built feedforward network used in CodinGame.
             Below, you can find a selection of packages I have used and models I have built that are more business related:
         """)
    st.write(" ")
    st.write(" - Tensorflow 2.0 with Keras")
    st.write(" - ANN for general deep learning classification or regression tasks")
    st.write(" - CNN for pattern recognition in images or as first layer in gameplay bots")
    st.write(" - RNN for timeSeries and NLP tasks")

with st.expander("Machine Learning Logistics Regression Playground", expanded=False):
    st.write(" Overall goal is to classify if patient has diabetes")

    def diabetes_positive(row_input):
        if row_input == "tested_positive":
            return 1
        return 0

    y = data_diabetes['target'].apply(diabetes_positive)
    X = data_diabetes['data']

    st.write(" This is how the raw data looks like. Target column has been removed (source is openml data_id=37) :")

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

    st.write("Model above is built as a very simple SKLearn pipeline with MinimaxScaler and Logisticregression.")
    st.write("I have also put a random state for response consistency. ")
