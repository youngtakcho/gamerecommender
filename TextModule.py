# this implementation is wrote after reading and learning sklearn's text module
# Reference
# Scikit-Learn. Sklearn text module in github.
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py

# import sklearn
import pickle
import sqlite3
# from gensim.models import word2vec , doc2vec , tfidfmodel
import gensim
import math
from gensim.parsing import *
import array
import numpy as np
from scipy import sparse as sp
from collections import defaultdict
from sklearn.externals import six
from sklearn.utils.validation import *
import numbers
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.preprocessing import normalize
import random

def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

class CountVectorizer:
    # dictionary = {"word": [total occurrence number of the word in all documents ,{d_id: word_counter}]}
    def __init__(self):
        self.vocab = None
        self.i_dict = {}
        self.word_dictionary = {}
        self.idx = 0
        self.num_of_doc = 0
        self.max_df = 1.0
        self.min_df = 1
        self.inverted_index_dict = {}
        self.X = None

    def addToDictionary(self,i_dict,word,d_id):
        if word not in i_dict:
            self.word_dictionary[word] = self.idx
            self.idx +=1
            self.i_dict[word] = [0, dict()]
        if d_id not in i_dict[word][1]:
            self.i_dict[word][1][d_id] = 0
        self.i_dict[word][1][d_id] += 1
        self.i_dict[word][0] += 1

    def buildInvertedIdx(self,data_rows):
        self.num_of_doc = len(data_rows)
        for idx,text in data_rows:
            words = self.preprocess(text)
            for w in words:
                self.addToDictionary(self.i_dict,w,idx)

    def getCounter(self,word):
        return self.i_dict[word][0]

    def getWords(self):
        return self.word_dictionary.keys()

    def preprocess(self,text):
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, stem_text, remove_stopwords,
                          strip_multiple_whitespaces, strip_short]
        return gensim.parsing.preprocess_string(text, CUSTOM_FILTERS)

    def _count_vocab(self,raw_docs,fixed_vocab=False,y=None):
        j_indices = []
        indptr = []
        if fixed_vocab:
            vocab = self.vocab
        else:
            vocab = defaultdict()
            vocab.default_factory = vocab.__len__
        values = array.array(str("i"))
        indptr.append(0)
        index = 0
        for raw_doc in raw_docs:
            feature_counter = {}
            doc = self.preprocess(raw_doc)
            for feature in doc:
                if y is not None:
                    if feature not in self.inverted_index_dict:
                        self.inverted_index_dict[feature] = set()
                    self.inverted_index_dict[feature].add(y[index])
                try:
                    feature_idx = vocab[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx]=0
                    feature_counter[feature_idx] +=1
                except KeyError:
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend((feature_counter.values()))
            indptr.append(len(j_indices))
            index += 1

        if not fixed_vocab:
            vocab = dict(vocab)

        indices_dtype = np.int64
        j_indices = np.asanyarray(j_indices,dtype=indices_dtype)
        indptr = np.asanyarray(indptr,dtype=indices_dtype)
        values = np.asanyarray(values,dtype=indices_dtype)
        X = sp.csr_matrix((values,j_indices,indptr),shape=(len(indptr) -1 ,len(vocab)),dtype=np.int64)
        X.sort_indices()
        if y is not None:
            self.X = X
        return vocab,X


    def _sort_features(self,X,vocab):
        sorted_feature = sorted(six.iteritems(vocab))
        map_index = np.empty(len(sorted_feature),dtype=X.indices.dtype)
        for new_val , (term , old_val) in enumerate(sorted_feature):
            vocab[term] = new_val
            map_index[old_val] = new_val
        X.indices = map_index.take(X.indices,mode='clip')
        return X

    def fit_transform(self,raw_docs,y=None):
        vocab , X = self._count_vocab(raw_docs,y=y)
        X = self._sort_features(X,vocab)
        n_doc = X.shape[0]
        max_doc_count = (self.max_df if isinstance(self.max_df,numbers.Integral) else self.max_df * n_doc)
        min_doc_count = (self.min_df if isinstance(self.min_df,numbers.Integral) else self.min_df * n_doc)
        if max_doc_count < min_doc_count:
            raise ValueError("Max and Min values of doc_count error Max < Min")
        self.vocab = vocab
        self.X = X
        return X

    def transform(self,raw_docs):
        _,X = self._count_vocab(raw_docs,fixed_vocab=True)
        return X


class TfIdfTransFormer():

    def __init__(self,norm='l2'):
        self.use_idf = True
        self.smooth_idf = 1.0
        self.sublinear_tf = False
        self.norm = norm
        return

    def fit(self, X):
        if not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples , n_feature = X.shape
            df = self._document_frequency(X).astype(dtype)
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)
            idf = np.log(n_samples/df) +1
            self._idf_diag = sp.diags(idf,offsets=0,
                                      shape=(n_feature,n_feature),
                                      format='csr',
                                      dtype=dtype)
        return self

    def transform(self,X,copy = True):
        X = check_array(X,accept_sparse=('csr','csc'),copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X,dtype=np.float64)
        n_samples, n_feature = X.shape

        if self.sublinear_tf:
            np.log(X.data,X.data)
            X.data += 1

        if self.use_idf:
            expected_n_features = self._idf_diag.shape[0]
            if n_feature != expected_n_features:
                raise ValueError("the number of features is not expacted number")
            X = X * self._idf_diag
        if self.norm:
            X = normalize(X,norm=self.norm,copy=False)

        return X

    def _document_frequency(self,X):
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices,minlength=X.shape[1])
        else:
            return np.diff(X.indptr)


class TfidfVectorizer(CountVectorizer):

    def __init__(self,norm='l2'):
        super().__init__()
        self._tfidf = TfIdfTransFormer()

    def fit(self,raw_documents,y=None):
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self,raw_docs,y=None):
        X = super().fit_transform(raw_docs)
        self._tfidf.fit(X)
        return self._tfidf.transform(X,copy=False)

    def transform(self,preprocessed_docs):
        X = super().transform(preprocessed_docs)
        return self._tfidf.transform(X,copy=False)


class Similarity:

    def calculate(self,X:sp.csr_matrix,q_X:sp.csr_matrix):
        q_X = normalize(q_X,norm="l1")
        sim = q_X * X.transpose()
        # sim = sim.transpose()
        return sim.toarray()[0]

class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.class_dictionary = {}
        self.n_doc_in_classes = {}
        self.total_words_in_class = {}
        self.class_word_probability = {}
        self.total_doc = 0
        return

    def fit(self,X,y):
        r_index = 0
        # sum all number of words for each classes.
        classes = defaultdict()
        classes.default_factory = classes.__len__
        for cls in y:
            class_id = classes[cls]
            if class_id not in self.class_dictionary:
                self.class_dictionary[class_id] = X.getrow(r_index)
                self.n_doc_in_classes[class_id] = 1
            else:
                self.class_dictionary[class_id] = self.class_dictionary[class_id] + X.getrow(r_index)
                self.n_doc_in_classes[class_id] += 1
            r_index += 1
        self.total_doc = r_index

        for cls in classes:
            self.total_words_in_class[cls] = self.class_dictionary[classes[cls]].sum()
            d_smoothing_factor = self.class_dictionary[classes[cls]].shape[1]
            Tf_vec = self.class_dictionary[classes[cls]].getrow(0).toarray() + 1  # for smoothing
            total_number_of_term = self.total_words_in_class[cls]
            p_words_g_class = Tf_vec / (total_number_of_term + d_smoothing_factor)  # for smoothing
            np.log(p_words_g_class)
            p_d = sp.lil_matrix((p_words_g_class.shape[1], p_words_g_class.shape[1]), dtype=np.float64)
            for i in range(0, p_words_g_class.shape[1]):
                p_d[i, i] = p_words_g_class[0][i]
            self.class_word_probability[cls] = p_d

        self.classes = dict(classes)

    def predict(self,X):
        predicted_results = []
        for i in range(0,X.shape[0]):
            target = X.getrow(i)
            results = []
            for cls in self.classes:
                cp_target = target.getrow(0)
                boolen_target = cp_target.getrow(0)
                boolen_target.data.fill(1)
                proba_in_class = self.class_word_probability[cls]
                proba_in_query = (boolen_target) * proba_in_class
                proba_in_query = proba_in_query.log1p()
                for i in range(0,len(proba_in_query.data)):
                    proba_in_query.data[i] *=  target.data[i]
                class_proba = self.n_doc_in_classes[self.classes[cls]] / self.total_doc
                results.append((proba_in_query.sum() + np.log1p(class_proba),cls))
            predicted_results.append(results)
        return predicted_results






if __name__ == "__main__":
    i_dic = {}
    conn = sqlite3.connect("game_data_modi.db")

    train_and_save = True
    if train_and_save:
        c = conn.cursor()
        c.execute("select r_id , text from reviews")
        rows = c.fetchall()
        y = [y for y, x in rows]
        X = [x for y, x in rows]
        start = time.time()
        print("start counting")
        wc = CountVectorizer()
        X_count = wc.fit_transform(X,y=y)
        tfidf = TfIdfTransFormer()
        X_tfidf = tfidf.fit(X_count)
        q = ["great music and beautiful background",]
        q_count = wc.transform(q)
        q_tfidf = tfidf.transform(q_count)
        print(q_tfidf)
        print("end counting , ",time.time() - start)

        with open("MyCountVectorizer.pickle","wb") as f:
            pickle.dump(wc,f)
        with open("MyTfIdfVectorizer.pickle", "wb") as f:
            pickle.dump(tfidf, f)
    else:
        with open("MyCountVectorizer.pickle","rb") as f:
            wc:CountVectorizer = pickle.load(f)
        with open("MyTfIdfVectorizer.pickle", "rb") as f:
            tfidf:TfidfVectorizer = pickle.load(f)

        load_from_db = True

        if load_from_db:
            dic2 = {}
            data_list = []
            conn = sqlite3.connect("game_data_modi.db")
            c = conn.cursor()
            q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 50 and length(text) > 100 and p_genre NOT like "%Action%" and p_genre NOT like "%Simulation%" and p_genre NOT like "%Indie%"  and p_genre not like "%Adventure%" and p_genre not like "RPG" """
            c.execute(q)
            rows = c.fetchall()
            for text, genres in rows:
                g = genres
                if g == "Animation &amp; Modeling" or g == "Design &amp; Illustration" or g == "None" or g == "Audio Production" \
                        or g == "Software Training" or g == "Video Production" or g == "Web Publishing" or g == "Utilities":

                    continue
                elif (g == "Free to Play" or g == "Early Access") and len(genres.split(",")) > 2:
                    g = genres
                dic = []
                dic.append(g)
                dic.append(text)
                if g not in dic2:
                    dic2[g] = 0
                dic2[g] += 1
                data_list.append(dic)

            c = conn.cursor()
            q = """select text,p_genre from reviews,products where reviews.product_id == products.id and hours > 150 and length(text) > 200 and p_genre like "Action" LIMIT 6000"""
            c.execute(q)
            rows = c.fetchall()
            for text, genres in rows:
                g = "Action"
                dic = []
                dic.append(g)
                dic.append(text)
                if g not in dic2:
                    dic2[g] = 0
                dic2[g] += 1
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

            X = [x for y, x in train_data]
            y = [y for y, x in train_data]



            wc_X = wc.transform(X)

            NB = NaiveBayes()
            NB.fit(wc_X,y)
            # import sys
            # sys.setrecursionlimit()
            with open("MyNaiveBayes.pickle", "wb") as f:
                pickle.dump(NB, f)
        else:
            with open("MyNaiveBayes.pickle", "rb") as f:
                NB = pickle.load(f)


        query = "beautiful and great background music"
        q_X = wc.transform([query])
        result = NB.predict(q_X)
        print(result)
