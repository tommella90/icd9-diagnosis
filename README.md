# SYMPHOMS TO ICD9 CODE
### Transform a medical diagnosis or symphoms into its ICD9 corresponding class. 

I use an SPACY embedding trained in the medical domain (available [here](https://allenai.github.io/scispacy/)) to transform medical text.  
This represents a part of a larger alghorythms we are using to help doctors to associate their diagnosis to the relative ICD9 code. 
**NB: This is not made to find the correct illness**, but to fast doctor work in linking the right class to their diagnose. 


#HOW:
#### 1) Assign Embedding to ICD9 
I assignet to each ICD9 description a vector (the model uses a 300 dimensions vector) 

#### 2) Assign a vector to the user input
Take an input (a list of sympthoms or a diagnosis) and assign it a value with the model

#### 3) Calculate the distance
By calculating the vectorial distance between the input and the ICD9 I can find the 'closest' ICD9 classes. In this app, I return the 5 most similar classes. 


