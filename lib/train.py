import pandas as pd
import numpy as np
import re
from sklearn  import preprocessing
import pickle
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction import text

fp_train = 'data/train.csv'
fp_val = 'data/val.csv'
fp_test = 'data/test.csv'

df = pd.read_csv(fp_val)

df['full_information']= df['title'] + ' ' + df['description']

def data_preprocessing(data):
    prep1 = data.lower()
    prep2 = re.sub('https?://\S+|www\.\S+', '', data)
    prep3 = re.sub('\\W', ' ', data)
    prep4 = re.sub('\n', '', data)
    prep5 = re.sub(' +', ' ', data)
    prep6 = re.sub('^ ', '', data)
    prep7 = re.sub(' $', '', data)
    return data

df['full_information'].apply(data_preprocessing)

stop={'а','без','более','больше','будет','будто','бы','был','была','были','было',
'быть','в','вам','вас','вдруг','ведь','во','вот','впрочем','все','всегда','всего','всех','всю','вы','где','да','даже',
'два','для','до','другой','его','ее','ей','ему','если','есть','еще','ж','же','за','зачем','здесь','и',
'из','или','им','иногда','их','как','какая','какой','когда','конечно','кто','куда','ли','лучше',
'меня','мне','много','может','можно','мой','моя',
'мы','на','над','надо','наконец','нас','не','него','нее','ней',
'нельзя','нет','ни','нибудь','никогда','ним','них','ничего',
'но','ну','о','об','один','он','она','они','опять','от','перед','по','под','после','потом','потому',
'почти','при','про','раз','разве','с','сам','свою','себе','себя','сейчас','со','совсем','так','такой','там','тебя','тем',
'теперь','то','тогда','того','тоже','только','том','тот','три','тут','ты','у','уж',
'уже','хорошо','хоть','чего','чем','через','что','чтоб','чтобы','чуть','эти','этого','этой','этом','этот','эту','я'}

def remove_stopwords(data):
    words = [word for word in data if word not in stop]
    words= "".join(words).split()
    words= [words.lower() for words in data.split()]
    return words    

df['full_information'].apply(remove_stopwords)

lemmatizer = WordNetLemmatizer()
def lemmatization(text):
    lemmas = []
    for word in text.split():
        lemmas.append(lemmatizer.lemmatize(word))
    return " ".join(lemmas)
df['full_information'].apply(lemmatization)

my_stop_words = text.ENGLISH_STOP_WORDS.union(["russian"])

my_tfidf = TfidfVectorizer(stop_words=my_stop_words, max_df=0.7)

ada = ADASYN(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(df['full_information'], df['is_bad'], test_size=0.3)
tfidf_train = my_tfidf.fit_transform(X_train)

tf_transformer = my_tfidf.fit(X_train)

tfidf_test = my_tfidf.transform(X_test) 
pickle.dump(tf_transformer, open("tf_idf.pkl", "wb"))
X_res, y_res = ada.fit_resample(tfidf_train, y_train)

classifier = LogisticRegression(random_state=42)
classifier.fit(X_res, y_res)

pickle.dump(classifier, open('model.pkl', 'wb'))