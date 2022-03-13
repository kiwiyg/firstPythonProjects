import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
#from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    original_input = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    original_input = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguinscleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
concat_input = pd.concat([original_input,penguins],axis=0) 
#is being concat-ed into the existing dataframe 
#so that datatype of all entries becomes byte so that functions 
#like prediction can work on the inputted entries because they only word on bytes format

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(concat_input[col], prefix=col)
    concat_input = pd.concat([concat_input,dummy], axis=1)
    del concat_input[col]
original_input = concat_input[:1] # Selects only the first row (the user input data)
#which has been converted into byte format due to step 46




# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(original_input)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(original_input)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(original_input)
prediction_proba = load_clf.predict_proba(original_input)

col2, col3 = st.columns((1,3))

col2.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
col2.write(penguins_species[prediction])

col3.subheader('Prediction Probability')
resulting_species=pd.DataFrame(prediction_proba)
resulting_species= resulting_species.rename({0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}, axis='columns')
col3.write(resulting_species)