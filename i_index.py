import sklearn
import sqlite3
from gensim.models import word2vec , doc2vec
import gensim
import json
from gensim.parsing import *
from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def preprocess(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,stem_text,remove_stopwords,strip_multiple_whitespaces,strip_short]
    return gensim.parsing.preprocess_string(text,CUSTOM_FILTERS)


conn = sqlite3.connect("game_data copy 3.db")

c = conn.cursor()

c.execute("select r_id , text from reviews")

rows = c.fetchall()
dictionary = {}
doctionary = {}
size_ = len(rows)
for (idx , text) in rows:
    id_list = []
    pre_processed = preprocess(text)
    for i in pre_processed:
        if i not in dictionary:
            dictionary[i] = set()
        dictionary[i].add(idx)
    if int(idx/size_*100) % 10 == 0:
        print(idx/size_*100,"%")

for i in dictionary:
    dictionary[i] = list(dictionary[i])
j_dic=json.dumps(dictionary)
with open("dictionary.json","w") as f:
    f.write(j_dic)
