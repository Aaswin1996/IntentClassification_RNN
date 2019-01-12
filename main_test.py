import os
import re
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GRU, BatchNormalization
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.preprocessing import OneHotEncoder
from spacy import lemmatizer
from sklearn.model_selection import train_test_split


def load_dataset(filename):
    df = pd.read_csv(filename, encoding="latin1", names=['Sentence', 'Intent'])
    intent = df['Intent']
    unique_intent = list(set(intent))
    sentences = list(df['Sentence'])

    return (intent, unique_intent, sentences)


def cleaning(sentences):
    words =[]
    for s in sentences:
        clean =re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        words.append([i.lower() for i in w])
    return words

intent , unique_intent,sentences = load_dataset("Dataset.csv")
cleaned_words = cleaning(sentences)

#creating tokenizer
def create_tokenizer(words,
                  filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token
#getting maximum length
def max_length(words):
  return(len(max(words, key = len)))
#encoding list of words
def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
print(encoded_doc)

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length,
                        padding =   "post"))
padded_doc = padding_doc(encoded_doc,max_length)
print(padded_doc)


output_tokenizer = create_tokenizer(unique_intent,
                        filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

encoded_output = encoding_doc(output_tokenizer, intent)
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
output_one_hot = one_hot(encoded_output)
#print(output_one_hot.shape)
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)
print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


def create_model(vocab_size, max_length):
    model = Sequential()

    model.add(Embedding(vocab_size, 128,
                        input_length=max_length, trainable=False))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(21, activation="softmax"))

    return model
if os.path.isfile("model.h5") == False:
    model = create_model(vocab_size, max_length)

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])


model = load_model("model.h5")


def predictions(text):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    print(test_word)
    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))

    test_ls = np.array(test_ls).reshape(1, len(test_ls))

    x = padding_doc(test_ls, max_length)

    pred = model.predict_proba(x)

    return pred


def get_final_output(pred, classes):
    predictions = pred[0]

    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)

    for i in range(pred.shape[1]):
        print("%s has confidence = %s" % (classes[i], (predictions[i])))

text = "How to borrow money?"
pred = predictions(text)
get_final_output(pred, unique_intent)

