from fastapi import FastAPI, Form
import uvicorn
import numpy as np
import pandas as pd
import time
#import scispacy
import scipy
from scipy.spatial import distance
import spacy


#%%
import pandas as pd
from scipy.spatial import distance
import spacy
print("Uploading the model")
sci_bio = spacy.load("en_ner_bionlp13cg_md")
#import en_ner_bionlp13cg_md
#sci_bio = en_ner_bionlp13cg_md.load()
print("Model uploaded successfully")


#%%
def string_to_array(string):
    char_to_remove = ['[', ']', '\n']
    for char in char_to_remove:
        string = string.replace(char, "")
    string = string.replace('  ', ' ')

    string = string.split(' ')
    while '' in string:
        string.remove('')
    array = np.array(string)
    array = array.astype('float64')                # converting to `float64`

    return array


def FindVectorsDistance(vector1, vector2):
    return scipy.spatial.distance_matrix([vector1], [vector2], threshold=1000)

def query_input(input, model, codes_vectorized, dataframe):
    # 1) translate the call from italian to english and assing a vector position
    query = model(input).vector

    # 2) calculate the distance from the call to all the codes in the tree
    list_distances = [float(FindVectorsDistance(query, code)) for code in codes_vectorized]
    dataframe['dist'] = list_distances

    # 3) retain the 5 closets matches
    sorted_codes = dataframe.sort_values(by=['dist'], ascending = True)
    return sorted_codes

def query_input_pandas(input, model_vectors, dataframe, n_matches = 10):
    ''' Function to translate the input call (medical text) into a list of tuple
    showing the 5 closest illnesses-related codes and their vectorial distances'''
    sorted_codes = query_input(input, sci_bio, model_vectors, dataframe)

    if n_matches == -1:
        n_matches = len(sorted_codes.index)

    top_matches = sorted_codes.iloc[0: n_matches, [0,1]]
    top_matches = top_matches.set_index('code')
    codes = list(top_matches.index)
    descriptions = list(top_matches.description)
    matches = dict(zip(codes, descriptions))

    return matches


#%%
data = pd.read_csv("codes_embedded.csv.gz", compression='gzip')
data['sci_bio_lower'] = data['sci_bio_lower'].apply(string_to_array)


#%%
app = FastAPI()
@app.get("/welcome")
async def hello():
    return "welcome to the medical app"

@app.post("/text/")
async def login(text: str = Form()):
    print("Write your medical text here: ")
    results = query_input_pandas(text, data['sci_bio_lower'], data, 5)
    return {"diagnosis": text, "RESULTS": results}
    #return text + " is the text you wrote"

if __name__ == "__main__":
    uvicorn.run(app, port=8888, host='127.0.0.1')
#http://127.0.0.1:8888/docs


#%%
