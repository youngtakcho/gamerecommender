import gensim
from gensim.parsing import *
import sqlite3
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.naive_bayes import *
from sklearn.svm import LinearSVC
import pickle



def preprocess(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,stem_text,remove_stopwords,strip_multiple_whitespaces,strip_short]
    return gensim.parsing.preprocess_string(text,CUSTOM_FILTERS)

def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def count_words(training_set):
    counts = defaultdict(lambda: [0, 0])
    for message, genre in training_set:
        for word in preprocess(message):
            counts[word][genre] += 1
    return counts


if __name__ == "__main__":

    data_list = []
    dic2 = {}


    conn = sqlite3.connect("game_data_modi.db")
    c = conn.cursor()
    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 50 and length(text) > 100 and p_genre NOT like "%Action%" and p_genre NOT like "%Simulation%" and p_genre NOT like "%Indie%"  and p_genre not like "%Adventure%" and p_genre not like "RPG" """
    c.execute(q)
    rows = c.fetchall()
    for text,genres in rows:
        g = genres
        if g == "Animation &amp; Modeling" or g == "Design &amp; Illustration" or g == "None" or g=="Audio Production" \
                or g == "Software Training" or g == "Video Production" or g == "Web Publishing" or g=="Utilities":

            continue
        elif (g == "Free to Play" or g == "Early Access") and len(genres.split(",")) >2:
            g = genres
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] +=1
        data_list.append(dic)

    c = conn.cursor()
    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "Action" LIMIT 6000"""
    c.execute(q)
    rows = c.fetchall()
    for text,genres in rows:
        g = "Action"
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] +=1
        data_list.append(dic)

    c = conn.cursor()
    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "Simulation" LIMIT 6000"""
    c.execute(q)
    rows = c.fetchall()
    for text, genres in rows:
        g = "Simulation"
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] += 1
        data_list.append(dic)
    c = conn.cursor()

    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "Indie" LIMIT 6000"""
    c.execute(q)
    rows = c.fetchall()
    for text, genres in rows:
        g = "Indie"
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] += 1
        data_list.append(dic)

    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "Adventure" LIMIT 6000"""
    c.execute(q)
    rows = c.fetchall()
    for text, genres in rows:
        g = "Adventure"
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] += 1
        data_list.append(dic)

    q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "RPG" LIMIT 6000"""
    c.execute(q)
    rows = c.fetchall()
    for text, genres in rows:
        g = "RPG"
        dic = []
        dic.append(g)
        dic.append(text)
        if g not in dic2:
            dic2[g] = 0
        dic2[g] += 1
        data_list.append(dic)

    print(dic2)

    tuples = [tuple(x) for x in data_list]
    data = tuples

    random.seed(0)
    train_data, test_data = split_data(data, 0.70)

    X = [x for y,x in train_data]
    y = [y for y,x in train_data]
    # count_vect = CountVectorizer(tokenizer=preprocess)
    count_vect = CountVectorizer(stop_words="english")
    X_train_counts = count_vect.fit_transform(X)

    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_counts, y)
    # clf = GaussianNB().fit(X_train_counts.toarray(), y)
    # clf = BernoulliNB().fit(X_train_counts.toarray(), y)
    #clf = LinearSVC().fit(X_train_tfidf,y)
    Xt = [x for y, x in test_data]
    yt = [y for y, x in test_data]
    X_new_counts = count_vect.transform(Xt)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    print(clf.score(X_new_counts,yt))
    with open("CountVectorizer.pickle", "wb") as f:
        pickle.dump(count_vect, f)
    with open("MultinomialNB.pickle", "wb") as f:
        pickle.dump(clf, f)


