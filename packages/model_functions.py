import time
import pandas as pd
import numpy as np
import tensorflow as tf
import re
# import nltk
# import ast
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import median_absolute_error as mae
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import accuracy_score as acc
# import matplotlib.pyplot as plt
#
# from keras.models import Sequential
# from keras import initializers
# from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense,  \
#                          BatchNormalization, Flatten, Reshape, Concatenate, add
# from keras.layers.recurrent import LSTM, GRU
# from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.models import Model
# from keras.optimizers import Adam, SGD, RMSprop
# from keras import regularizers

from tensorflow.keras.layers import concatenate, add

from tensorflow.keras.utils import plot_model
max_daily_length = 100

def news_to_int(news,vocab_to_int):
    '''Convert your created news into integers'''
    ints = []
    for word in news.split():
        if word in vocab_to_int:
            ints.append(vocab_to_int[word])
        else:
            ints.append(vocab_to_int['<UNK>'])
    return ints


def padding_news(news,vocab_to_int):
    '''Adjusts the length of your created news to fit the model's input values.'''
    padded_news = news
    if len(padded_news) < max_daily_length:
        for i in range(max_daily_length-len(padded_news)):
            padded_news.append(vocab_to_int["<PAD>"])
    elif len(padded_news) > max_daily_length:
        padded_news = padded_news[:max_daily_length]
    return padded_news


def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    price = price*(max_price-min_price)+min_price
    return(price)

def predict(create_news, model,vocab_to_int,contractions):
    clean_news = clean_text(create_news,contractions)

    int_news = news_to_int(clean_news,vocab_to_int)

    pad_news = padding_news(int_news,vocab_to_int)
    pad_news = np.array(pad_news).reshape((1, -1))

    pred = model.predict([pad_news])
    price_change = unnormalize(pred)

    print("The Nifty should open: {} from the previous open.".format(np.round(price_change[0][0], 2)))


def clean_text(text,contractions, remove_stopwords=True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'0,0', '00', text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)

    # Optionally, remove stop words
    # if remove_stopwords:
    #     text = text.split()
    #     stops = set(stopwords.words("english"))
    #     text = [w for w in text if not w in stops]
    #     text = " ".join(text)

    return text