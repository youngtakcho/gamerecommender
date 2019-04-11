import sqlite3
import gensim
import math
from gensim.parsing import *
import pickle


class CountVectorizer:
    # dictionary = {"word": [total occurrence number of the word in all documents ,{d_id: word_counter}]}
    def __init__(self):
        self.i_dict = {}
        self.word_dictionary = {}
        self.idx = 0
        self.num_of_doc = 0

    def addToDictionary(self,i_dict,word,d_id):
        if word not in i_dict:
            self.word_dictionary[word] = self.idx
            self.idx +=1
            self.i_dict[word] = [0, dict()]
        if d_id not in i_dict[word][1]:
            self.i_dict[word][1][d_id] = 0
        self.i_dict[word][1][d_id] += 1
        self.i_dict[word][0] += 1

    def fit(self,data_rows):
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

class TfIdf:
    def __init__(self,word_counter):
        self.word_counter:CountVectorizer = word_counter

    def getTfIdf(self,sentence):
        tf_dict = {}
        words = self.word_counter.preprocess(sentence)
        for w in words:
            if w not in tf_dict:
                tf_dict[w] = 0;
            tf_dict[w] += 1;
        total_words = len(tf_dict)
        tfidf_dict = {}
        for k in tf_dict:
            tfidf_dict[k] = math.log10((tf_dict[k]/total_words)) * self.getIdf(k)
        return tfidf_dict, tf_dict

    def getIdf(self,word):
        n_t = len(self.word_counter.i_dict[word][1])
        idf = 0. + self.word_counter.num_of_doc / n_t
        return math.log10(idf)

    def vectorize(self,sentence):
        tfidf_dict, tf_dict = self.getTfIdf(sentence)
        vec = dict()
        for k in tf_dict:
            idx = self.word_counter.word_dictionary[k]
            vec[idx] = tfidf_dict[k]
        return vec


# class NaiveBayesclassifier:
#

if __name__ == "__main__":
    i_dic = {}
    conn = sqlite3.connect("game_data_modi.db")
    c = conn.cursor()
    c.execute("select r_id , text from reviews")
    rows = c.fetchall()
    wc = CountVectorizer()
    wc.fit(rows)
    tfidf = TfIdf(word_counter=wc)
    print(tfidf.vectorize("good game for every one"))
    with open("mywordcount.pickle", "wb") as f:
        pickle.dump(wc, f)
    # with open("mytfidf.pickle", "wb") as f:
    #     pickle.dump(tfidf, f)
