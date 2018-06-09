# Comparison of performance of LSTM + word/character embeddings with CRF
# David Di Valentino, 2018
# Shamelessly adopting LSTM training code from:
# https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/

import pandas as pd
import numpy as np

from ner_train_crf import ner_train_crf

# Print the classification report.
from sklearn.metrics import classification_report

# For parsing the classification report.
from parse_class_report import parse_class_report

import sys

################################################################################
# Some configurables.
################################################################################

trainLSTM=False
trainCRF=True

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
n_words = len(words); n_words
tags = list(set(data["Tag"].values))
n_tags = len(tags); n_tags

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter    = SentenceGetter(data)
sentences = getter.sentences

max_len = 75
max_len_char = 10

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

from keras.preprocessing.sequence import pad_sequences
X_word = [[word2idx[w[0]] for w in s] for s in sentences]

X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print(n_chars)

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

# Define the label vector for later classification reporting.
labels = []
for key, value in sorted(tag2idx.iteritems(), key=lambda (k,v): (v,k)):
    if key is None :
        labels.append('None')
    if key is 'PAD' :
        continue;
    else :
        labels.append(key)

###########################################################################
# Insert CRF training here. Save the model, and return arrays of 
# x_test, y_test, and y_true.
###########################################################################

if trainCRF :
    x_test, y_pred, y_true = ner_train_crf(sentences);
    
    # Convert the sequences of tags to a 1D array.
    predictions = np.array([tag2idx[tag] for row in y_pred for tag in row])
    truths      = np.array([tag2idx[tag] for row in y_true for tag in row])

    # Print the classification report.
    class_report_crf = classification_report(
        truths,predictions,
        target_names=labels
    )

    crf_labels,crf_precision,crf_recall,crf_f1,crf_support = parse_class_report(class_report_crf);
    
X_char = []
for sentence in sentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

from sklearn.model_selection import train_test_split

X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D

if trainLSTM :

    # Word input + embeddings.
    word_in = Input(shape=(max_len,))
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                        input_length=max_len, mask_zero=True)(word_in)

    # Character input + embeddings.
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                            input_length=max_len_char, mask_zero=True))(char_in)
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                    recurrent_dropout=0.5))(emb_char)

    # Build the LSTM.
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)

    model = Model([word_in, char_in], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.summary()
    history = model.fit([X_word_tr,
                        np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                        np.array(y_tr).reshape(len(y_tr), max_len, 1),
                        batch_size=32, epochs=10, validation_split=0.1, verbose=1)
    model.save('lstm_char_embedding.h5')

# Load (or reload) the model.
print("Evaluating LSTM model ...");
from keras.models import load_model
model = load_model('lstm_char_embedding.h5')

# Apply to the testing sample.
y_pred = model.predict([X_word_te,
                        np.array(X_char_te).reshape((len(X_char_te),
                        max_len, max_len_char))])

# Prepare the classification report inputs.
y_te_flat=[]
y_pred_flat=[]
for i in range(0,len(y_te)) :
    p = np.argmax(y_pred[i], axis=-1)
    for j in range(0,len(y_te[0])) :
        y_pred_flat.append(p[j])
        if y_te[i][j]<1 :
            y_te_flat.append(7)
        else :
            y_te_flat.append(y_te[i][j])

# Generate a classification report for the LSTM and compare to the
# performance of the CRF.
class_report_lstm = classification_report(
    y_te_flat,y_pred_flat,
    target_names=labels
)
lstm_labels,lstm_precision,lstm_recall,lstm_f1,lstm_support = parse_class_report(class_report_lstm);

for i in range(0,len(lstm_labels)) :
    print("%10s : %11s , %11s" % ("Label","CRF","LSTM"));
    print("%10s : %6.5f , %6.5f" % (lstm_labels[i],crf_recall[i],lstm_recall[i]))
