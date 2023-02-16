import pandas as pd
import numpy as np
import nltk

nltk.download("stopwords")

print("Uploading the model")
import en_ner_bionlp13cg_md
sci_bio = en_ner_bionlp13cg_md.load()
print("Model uploaded successfully")


#%% Apply embedding model to ill descriptions
data = pd.read_excel("codes_icd9.xlsx")
data.columns = ['code', 'description', 'short_description']

# calculate vec. positions for every code
list_nlp_eng_bio_lower = []

for text in data['description']:
    text = text.lower()
    doc_bio = sci_bio(text)
    vec_bio = doc_bio.vector
    list_nlp_eng_bio_lower.append(vec_bio)

data['sci_bio_lower'] = list_nlp_eng_bio_lower

#data.to_pickle('codes_embedded.pkl')
#data.to_csv('dati/codes_embedded.txt', sep=' ')
#data = pd.read_pickle("../codes_embedded.pkl")
#data = data.iloc[0:100, :]
#data.to_pickle('codes_embedded.pkl')

data.to_csv("codes_embedded.csv.gz",
          index=False,
          compression="gzip")

print('done')
