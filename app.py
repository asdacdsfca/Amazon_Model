import streamlit as st
import time
import argparse
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import string
from sklearn.feature_extraction import _stop_words

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('1da').setLevel(logging.WARNING)

# Sample has already been cleaned in the notebook. To see the process, please refer to the notebook.
sample = pd.read_csv('sample.csv', header = 0, encoding = 'unicode_escape')
sample['general_cate'] = sample['general_cate'].str.lower()
sample = sample.dropna(subset=['Selling Price', 'general_cate', \
    'weights_ounces'])[['Selling Price', 'general_cate', 'words_len', 'weights_ounces']]
X = sample.drop(columns = ['Selling Price'])
y = sample['Selling Price']

def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    preproc = ColumnTransformer(
        transformers = [
            ('cat_cols', OneHotEncoder(sparse=False, handle_unknown='ignore'), \
                ['general_cate', 'weights_ounces']),
            ('function', FunctionTransformer(lambda x: np.log(x+1)), ['words_len'])
        ],
        remainder='passthrough'
    )
    pl = Pipeline([
        ('preprocessor', preproc),
        ('forest', RandomForestRegressor())
    ])
    pl.fit(X, y)
    return pl

def cleanWords(text):
    try:
        text = text.lower()
        #get out all the punctuations
        text = text.strip(string.punctuation)
        pattern = 'r[\r\t\n]*'
        clean = re.sub(pattern,' ', text)
        # Don't want stop words cause not relate to the topic
        words= [i for i in clean.split(' ')\
               if not i in stopwords.words('english')]
        return len(words)
    except:
        return 0

def convert(ounces):
    try:
        ounces = float(ounces)
        if 15>ounces>0:
            return '0-14'
        elif 30>ounces>15:
            return '0-29'
        elif 45>ounces>30:
            return '30-44'
        elif 70>ounces>45:
            return '45-69'
        elif 85>ounces>70:
            return '70-84'
        elif 100>ounces>85:
            return '85-99'
        elif 115>ounces>100:
            return '100-114'
        elif ounces>115:
            return '114-'
    except:
        return False

def writer():
    st.markdown(
        """
        # Prediction Model 
        ## (Reached a score of ~0.7 on Unseen Data)
        """

    )
    st.sidebar.subheader("Setting")
    generate_max_len = st.sidebar.number_input("generate_max_len", min_value = 0, max_value = 512, value=32)
    temperature = st.sidebar.number_input("temperature", min_value = 0.0, max_value = 100.0, value = 1.0)

    st.sidebar.subheader("Note")
    st.sidebar.subheader("*All the punctuations or stop words are not taken into consideration")
    st.sidebar.subheader("*Haven't performed searching for the best hyperparameters yet, too slow :(")

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_max_len', default=generate_max_len, \
        type=int, help='Max length of generated title')
    parser.add_argument('--temperature', type=float, default=temperature, help='Max legnth of temp')
    args = parser.parse_args()

    cat = st.text_area("write your category here", max_chars = 512)
    title = st.text_area("write your product description here", max_chars = 512)
    weights_ounces = st.text_area("write your weights (in ounces) here", max_chars = 512)
    
    in_ounces = convert(weights_ounces)
    df_dict = {
    'general_cate': [cat.lower()],
    'words_len': [cleanWords(title)],
    'weights_ounces': [convert(weights_ounces)]
    }
    if st.button("click here to generate output"):
        start_message = st.empty()
        start_message.write("Initializing...")
        start_time = time.time()
        end_time = time.time()
        start_message.write("Completed, took {}s".format(end_time - start_time))
        if convert(weights_ounces) == False:
            result = 'Please try again! (hint: enter a number for weights)'
        else:
            result = random_forest(X, y).predict(pd.DataFrame(df_dict))[0]
        st.text_area("Generating Output", value = result, key = None)
    else:
        st.stop()

if __name__ == '__main__':
    writer()