from socket import sethostname
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
# df= df.rename({0:'setosa', 1:'versicolor', 2:'virginica'}, axis='columns')
# st.write(df.head())

st.subheader('User Input parameters')
#df= df.rename(columns = {'index':'species'})
st.write(df)

iris = datasets.load_iris()
irisdf= pd.DataFrame(data= np.c_[iris['data'], iris['target' ]], columns= iris['feature_names'] + ['species'])

for i in range(len(irisdf)):
    if   irisdf.iloc[i, 4]==0:
         irisdf.iloc[i, 4]= 'setosa' 
    elif irisdf.iloc[i, 4]==1:
         irisdf.iloc[i, 4]= 'versicolor'
    elif irisdf.iloc[i, 4]==2:
         irisdf.iloc[i, 4]= 'virginica' 
    i=i+1
#irisdf = pd.DataFrame(datasets.load_iris())
#irisdf= irisdf.rename({0:'setosa', 1:'versicolor', 2:'virginica'}, axis='columns')
st.write(irisdf)
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# target=pd.DataFrame(iris.target_names)
# target= target.rename({0:"species"}, axis='columns')
# st.write(target)

st.subheader('Prediction')
target=pd.DataFrame(iris.target_names[prediction])
target= target.rename({0:"species"}, axis='columns')
st.write(target)
#st.write(prediction)

st.subheader('Prediction Probability')
target=pd.DataFrame(prediction_proba)
target= target.rename({0:'setosa', 1:'versicolor', 2:'virginica'}, axis='columns')
st.write(target)




#tutorial's code for reference

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# clf = RandomForestClassifier()
# clf.fit(X, Y)

# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(iris.target_names)

# st.subheader('Prediction')
# st.write(iris.target_names[prediction])
# #st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)