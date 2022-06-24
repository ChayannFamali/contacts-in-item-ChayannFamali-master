from typing import Tuple, Union
import re
import pickle
import pandas as pd
from sklearn import preprocessing
from train import data_preprocessing, remove_stopwords, lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import numpy as np
def task1(df):
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    tf1 = pickle.load(open("tf_idf.pkl", 'rb'))
    df['description']= df['title'] + ' ' + df['description']
    df['description'].apply(data_preprocessing)
    df['description'].apply(remove_stopwords)
    df['description'].apply(lemmatization)
    my_stop_words = text.ENGLISH_STOP_WORDS.union(["russian"])
    my_tfidf = TfidfVectorizer(stop_words=my_stop_words, max_df=0.7,vocabulary=tf1.vocabulary_)
    tfidf_all = my_tfidf.fit_transform(df['description'])
    y_pred = clf.predict_proba(tfidf_all)
    prediction = pd.Series(y_pred[:,1])
    #index = pd.Series(range(0,len(y_pred)))
    #target_prediction = pd.DataFrame({"index":index, "prediction":prediction})
    return prediction

def task2(df):
    start = list()
    final = list()
    index = list(df.index)
    regex_phone = re.compile(r"([8-9]\d{7,10})|(\+7\d{7,10})|((\d.){8,11})|(\+7 \d{3})|(8[(-]\d{3})|(89 )|([8-9] \d)")
    regex_mess = re.compile(r"(vk.com)|(Discord)|(What's app)|(Whats app)|(Whatsapp)|(вотсап)|(вацап)|(viber)|(вайбер)")
    regex_email = re.compile(r"(http)|(@mail)|(@yandex)|(@yahoo)|(@gmail)|(@ya)|(@list)|(@bk)|(@outlook)")
    regex_email_2 = re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
    for i in range(len(df['description'])):
        m = re.search(regex_phone or regex_mess or regex_email or regex_email_2, df['description'][i])
        try:
            starts = m.start()
            finals = m.end()
        except AttributeError:
            starts = np.nan
            finals = np.nan
        start.append(starts)
        final.append(finals)
    #task_2 = pd.DataFrame(list(zip(index, starts, finals)), columns = ['index', 'start', 'final'])
    return start,final