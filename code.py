
# Author: TOSHAN MAJUMDAR
# Semi Supervised Learning for Text Classification (Natural Language Processing)
# Dataset Used: Kaggle IMDB movie reveiws : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download

import pandas as pd
import numpy as np
import sklearn
import nltk

##############################################################
#READING DATASET
ds = pd.read_csv('IMDB_Dataset.csv').sample(n=200)
##############################################################
# DATA CLEANING - Removing numbers and punctuations
import re #import the regular expression package, re
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    cleantext = re.sub("[^a-zA-Z]", " ", cleantext)
    cleantext = re.sub(' +', ' ', cleantext)
    return cleantext.lower()

ds['review'] = ds['review'].apply(lambda x: cleanhtml(x))
###############################################################
#STOP WORD REMOVAL
from nltk.corpus import stopwords # Import the stop word list
def tokenize(x):
    x = x.split()
    x = [i for i in x if i not in stopwords.words("english")]
    return x

ds['words'] = ds['review'].apply(lambda x: tokenize(x))

#Splitting dataset into training data, test data
from sklearn.model_selection import train_test_split
train, test = train_test_split(ds, test_size=0.2)

train_words = train['review'].values
train_sentiment = train['sentiment']
test_words = test['review'].values
test_sentiment = test['sentiment']

###############################################################
#FEATURE SELECTION
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",  tokenizer = None,  preprocessor = None,  stop_words = None,  max_features = 4000) #Bag of Words using CountVectorizer

train_features = vectorizer.fit_transform(train_words)
test_features = vectorizer.transform(test_words)

###############################################################
# CLASSIFIER SELECTION:

#(1)Random forest classifier
from sklearn.ensemble import RandomForestClassifier #import Library 
rf_estimator = RandomForestClassifier() #initiate the RandomForestClassifier
rf_estimator.fit(train_features, train_sentiment) # Training RFC using BOW
rf_result = rf_estimator.predict(test_features) # .predict() to make label predictions


#(2) Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB #import Library
gnb_estimator = GaussianNB() #initiate the GaussianNB classifier
gnb_estimator.fit(train_features.toarray(),train_sentiment) #training GNB using BOW
gnb_result = gnb_estimator.predict(test_features.toarray()) # .predict() to make label predictions


#(3) Logistic Regression
from sklearn.linear_model import LogisticRegression as LR #import Library
lr = LR() # initiate the LR classifier
lr.fit(train_features,train_sentiment) # Training LR using BOW
lr_result = lr.predict(test_features.toarray()) # .predict() to make label predictions

###################################################################
# STORING PREDICTED SENTIMENTS TO CSV FILE
rf_result = [str(i) for i in rf_result]  ;  gnb_result = [str(i) for i in gnb_result]  ;   lr_result = [str(i) for i in lr_result]
test_data = pd.DataFrame({'review': test_words, 'sentiment': test_sentiment, 'rfc_prediction':rf_result, 'gnb_prediction':gnb_result, 'lr_prediction':lr_result}, columns=['review', 'sentiment','rf_prediction','gnb_prediction','lr_prediction'])
test_data['rf_prediction'] = rf_result   ;  test_data['gnb_prediction'] = gnb_result   ;  test_data['lr_prediction'] = lr_result
test_data.to_csv(r'IMDB_Predict.csv', index = False, header=True)

############################## END ################################
