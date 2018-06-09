from sklearn.model_selection import train_test_split
import pycrfsuite

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm

from os.path import isfile

########################################################################
# Creating feature vectors for words.
########################################################################

def word2features(doc,pos,i) :
    
    if doc[i] == None :
        return [];
    
    word   = doc[i]
    postag = pos[i]

    # Common features for all words
    features = [
        'bias',
        'word.lower='+word.lower(),
        'word[-3:]='+word[-3:],
        'word[-2:]='+word[-2:],
        'word.isupper=%s'%word.isupper(),
        'word.istitle=%s'%word.istitle(),
        'word.isdigit=%s'%word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not at the beginning
    # of a document
    if i > 0 :
        word1   = doc[i-1]
        postag1 = pos[i-1]
        features.extend([
            '-1:word.lower='+word1.lower(),
            '-1:word.istitle=%s'%word1.istitle(),
            '-1:word.isupper=%s'%word1.isupper(),
            '-1:word.isdigit=%s'%word1.isdigit(),
            '-1:postag='+postag1
        ])
    else :
        # Indicate the beginning of the document
        features.append('BOS')

    # Features for words before the end of a document
    if i < len(doc)-1 :
        word1   = doc[i+1]
        postag1 = pos[i+1]
        features.extend([
            '+1:word.lower='+word1.lower(),
            '+1:word.istitle=%s'%word1.istitle(),
            '+1:word.isupper=%s'%word1.isupper(),
            '+1:word.isdigit=%s'%word1.isdigit(),
            '+1:postag='+postag1
        ])
    else :
        # End of the document
        features.append('EOS')
    
    return features

########################################################################
# Training a CRF using some input sentences with form (word,pos,tag).
########################################################################

def ner_train_crf (sent_pos_tags) :
    x_test = []
    y_pred = []
    y_true = []

    X_features = [];
    y          = [];

    print "Producing feature vectors ..."
    for isi in tqdm(range(0,len(sent_pos_tags))) :
        
        sentence = sent_pos_tags[isi];

        sent = []
        pos  = []
        tag  = []
        for word in sentence :
            sent.append(word[0]);
            pos.append(word[1]);
            tag.append(word[2]);
        
        # Create the feature vector for each word.
        X_feature = []
        for j in range(0,len(sentence)) :
            X_feature.append(word2features(sent,pos,j))
        X_features.append(X_feature)

        y.append(tag);

    # Split up into training, testing samples.
    # Split into training, testing data.
    X_train, X_test, y_train, y_test = train_test_split(X_features,y,test_size=0.1,random_state=2018)

    # Create the CRF model.
    if not isfile("crf.model") :
        
        print "Training the CRF model ..."
        trainer = pycrfsuite.Trainer(verbose=True)
        for xseq, yseq in zip(X_train, y_train) :
            trainer.append(xseq,yseq)

        # Set the parameters of the model
        trainer.set_params({
            # coefficient for L1 penalty
            'c1': 0.1,
            # coefficient for L2 penalty
            'c2': 0.01,  
            # maximum number of iterations
            'max_iterations': 200,
            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })

        # Train the model.
        trainer.train('crf.model')

    # Apply the model to the testing data.
    tagger = pycrfsuite.Tagger(verbose=True)
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    y_true = y_test;

    return (X_test,y_pred,y_true);
