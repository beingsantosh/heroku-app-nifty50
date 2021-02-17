# from modules.util_functions import
import json
import numpy as np
import re

import os


contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=False):
    """Remove unwanted characters and format the text to create fewer nulls word embeddings"""

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


def unnormalize_data(data, max_price, min_price):
    """Revert values to their unnormalized amounts"""
    unnormalized = []
    for d in data:
        unnormalized.append(d * (max_price - min_price) + min_price)
    return unnormalized

def read_json(filename):
    filename = './predictions/' + filename

    if filename.split('.')[-1] != 'json':
        filename = filename + '.json'

    with open(filename, 'r') as keyfile:
        # data = keyfile.read()
        obj = json.loads(keyfile.read())
    return obj


def news_to_int(news, filename):
    """Convert your created news into integers"""
    ints = []
    vocab_to_int = read_json(filename)
    for word in news.split():
        if word in vocab_to_int:
            ints.append(vocab_to_int[word])
        else:
            ints.append(vocab_to_int['<UNK>'])
    return ints


def padding_news(news, filename, max_daily_length):
    """Adjusts the length of your created news to fit the model's input values."""
    vocab_to_int = read_json(filename)
    padded_news = news
    if len(padded_news) < max_daily_length:
        for i in range(max_daily_length - len(padded_news)):
            padded_news.append(vocab_to_int["<PAD>"])
    elif len(padded_news) > max_daily_length:
        padded_news = padded_news[:max_daily_length]
    return padded_news


def predict(create_news, model, vocab_filepath, max_daily_length):
    """

    :param create_news:
    :param model:
    :param vocab_filepath:
    :param max_daily_length:
    :return:
    """

    clean_news = clean_text(create_news)

    # vocab_to_int = read_json(vocab_filepath)

    int_news = news_to_int(clean_news, vocab_filepath)

    # print(len(int_news))
    pad_news = padding_news(int_news, vocab_filepath, max_daily_length)
    # print((pad_news))
    pad_news = np.array(pad_news).reshape((1, -1))

    pred = model.predict([pad_news])
    # print(pad_news.shape)
    dic_max_min = read_json('normalize_metadata')
    price_change = unnormalize_data(pred, dic_max_min['max_price'], dic_max_min['min_price'])

    print("The Nifty should open: {} from the previous open.".format(price_change[0][0]))
    return price_change[0][0]
