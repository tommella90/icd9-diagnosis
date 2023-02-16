import streamlit as st
import numpy as np
import pandas as pd
import time
#import scispacy
import scipy
from scipy.spatial import distance
import spacy

## UPLOAD AND CLEAN DATA
print("Uploading the model")
sci_bio = spacy.load("en_ner_bionlp13cg_md")
print("Model uploaded successfully")
data = pd.read_pickle("codes_embedded.pkl")

def string_to_array(string):
    char_to_remove = ['[', ']', '\n']
    for char in char_to_remove:
        string = string.replace(char, "")
    string = string.replace('  ', ' ')

    string = string.split(' ')
    while '' in string:
        string.remove('')
    array = np.array(string)
    array = array.astype('float64')

    return array

data['sci_bio_lower'] = data['sci_bio_lower'].apply(string_to_array)


def FindVectorsDistance(vector1, vector2):
    return scipy.spatial.distance_matrix([vector1], [vector2], threshold=1000)

def query_input(input, model, codes_vectorized):
    # 1) translate the call from italian to english and assing a vector position
    query = model(input).vector

    # 2) calculate the distance from the call to all the codes in the tree
    list_distances = [float(FindVectorsDistance(query, code)) for code in codes_vectorized]
    data['dist'] = list_distances

    # 3) retain the 5 closets matches
    sorted_codes = data.sort_values(by=['dist'], ascending = True)
    return sorted_codes

def query_input_pandas(input, model_vectors, n_matches = 10):
    ''' Function to translate the input call (medical text) into a list of tuple
    showing the 5 closest illnesses-related codes and their vectorial distances'''
    start_time = round(time.time(), 2)
    sorted_codes = query_input(input, sci_bio, model_vectors)

    if n_matches == -1:
        n_matches = len(sorted_codes.index)

    top_matches = sorted_codes.iloc[0: n_matches, [0,1]]
    top_matches = top_matches.set_index('code')
    codes = list(top_matches.index)
    descriptions = list(top_matches.description)
    matches = dict(zip(codes, descriptions))

    #print(top_matches)
    #print("TIME", "--- %s seconds ---" % (round(time.time(), 2) - start_time))

    return top_matches


st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="ICD9 - DIAGNOSIS",
                   page_icon=":🧊:")

with st.container():
    st.title("ICD9 DIAGNOSIS APP")
    st.subheader("Get the ICD9 code for your diagnosis")
    text = st.text_input('Write a diagnosis')
    results = query_input_pandas(text, data['sci_bio_lower'], 5)
    #d = {"diagnosis": text, "RESULTS": results}
    st.write("Diagnosis", text)
    st.write(results)


#%%
