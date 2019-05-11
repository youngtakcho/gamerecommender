import sqlite3
import gensim
import pickle
from gensim.parsing import *
# from gensim.similarities.docsim import Similarity
from flask import Flask , render_template , request , url_for
from flask_table import Table, Col , LinkCol
from sklearn.externals import joblib
import copy
import time
from TextModule import *


def preprocess(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,stem_text,remove_stopwords,strip_multiple_whitespaces,strip_short]
    return gensim.parsing.preprocess_string(text,CUSTOM_FILTERS)

app = Flask(__name__)

# app.dictionary = gensim.corpora.Dictionary.load("./dictionary.gensim")

# app.data = None
print("load libs")


# app.count_vect = joblib.load("MyCountVectorizer.pickle")
# app.tfidf_transformer = joblib.load("MyTfIdfVectorizer.pickle")
# app.clf = joblib.load("LinearSVC.pickle")

# app.clf = joblib.load("MyNaiveBayes.pickle")
app.tfidf_use = False
print("load dic")
# with open('dictionary.pickle', 'rb') as handle:
#     app.data = pickle.load(handle)


@app.route('/')
def main_post(query=None):
    html = """<H3>Game Recommender</H3>
<form action="/search" method="post"><input name="query" type="text" /><button formmethod="post" type="submit">Submit</button></form>"""
    return render_template('index.html')

@app.route("/search",methods=['GET','POST'])
def search():
    start = time.time()

    if request.method == "GET":
        query = request.args.get('query')
    elif request.method == "POST":
        query = request.form["query"]
    docs_new = [query]
    X_new_counts = app.count_vect.transform(docs_new)

    X_new_tfidf = app.tfidf_transformer.transform(X_new_counts)

    predicted = app.clf.predict(X_new_counts)[0]

    query_arr = preprocess(query)
    results = set()

    try:
        results = app.count_vect.inverted_index_dict[query_arr[0]]
    except KeyError as e:
        print("no key")
    for i in query_arr[1:]:
        try:
            results = results.intersection(app.count_vect.inverted_index_dict[i])
        except KeyError as e:
            continue


    conn = sqlite3.connect("game_data_modi.db")
    c = conn.cursor()
    q = "SELECT r_id,text from reviews WHERE r_id IN " + str(tuple(results)) + ""
    c.execute(q)
    rows = c.fetchall()

    bows = []
    indices = []
    for idx, review in rows:
        # pre = preprocess(review)

        bows.append(review)
        indices.append(idx)
    bows = app.count_vect.transform(bows)
    bows = app.tfidf_transformer.transform(bows)
    siml = Similarity()
    result_siml = siml.calculate(bows,X_new_tfidf)
    # siml = Similarity(None, bows, num_features=len(dictionary))
    # result_siml = siml[bow_q]

    ordered = sorted(range(len(result_siml)), key=lambda k: result_siml[k],reverse=True)
    r_list = []
    for i in ordered:
        r_list.append(indices[i])  # list of reviews.
    pre_ordered = sorted(range(len(predicted)), key=lambda k: predicted[k][0],reverse=True)
    pre_list = []
    for i in pre_ordered:
        pre_list.append(predicted[i])  # list of reviews.


    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # sql_query = "select product_id from reviews where r_id = "
    d_pr = {}
    product_set = set()
    for i in r_list:
        c.execute("select product_id from reviews where r_id = " + str(i))
        rows = c.fetchall()
        for r in rows:
            if r[0] is None or type(r[0]) == str:
                print("None")
                continue
            if r[0] not in d_pr:
                d_pr[r[0]] = [i]
            else:
                d_pr[r[0]].append(i)
            product_set.add(r[0])
    t_rows = []
    classified_set = set()
    for pro in product_set:
        q = "select title,developer,genres,tags from products WHERE id=" + str(pro) +""" and genres like "%"""+str(pre_list[0][1])+"""%\""""
        print(q)
        c.execute(q)
        rows = c.fetchall()
        for title, developer, tags, genres in rows:
            classified_set.add(pro)
            link_str = ""
            for r_id in d_pr[pro][:5]:
                link_str += """<a href=\"/review?r_id="""+str(r_id)+"""\" target=\"_blank\">"""+str(r_id)+""",</a>"""
            t_rows.append(Item(title, developer, tags, genres,link_str))
    product_set = product_set-classified_set

    for pro in product_set:
        c.execute("select title,developer,genres,tags from products WHERE id=" + str(pro))
        rows = c.fetchall()
        for title, developer, tags, genres in rows:
            link_str = ""
            for r_id in d_pr[pro][:5]:
                link_str += """<a href=\"/review?r_id="""+str(r_id)+"""\" target=\"_blank\">"""+str(r_id)+""",</a>"""
            t_rows.append(Item(title, developer, tags, genres,link_str))

    table = ItemTable(t_rows)
    html = str(table.__html__())
    html = html.replace("&#34;","\"")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    end = time.time()
    print(end - start)
    html_str = "pridict results : <br>-----------------<br>"
    for i in range(0,len(pre_list)):
        html_str += str(pre_list[i][1]) + " : " + str(pre_list[i][0]) + "<br>"
    html_str += html
    conn.close()
    return html_str

@app.route("/review",methods=['GET','POST'])
def show_review():
    if request.method == "GET":
        query = request.args.get('r_id')
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()
    sql_q = "select text , product_id , username from reviews where r_id="+str(query)
    c.execute(sql_q)
    row = c.fetchone()
    r_text , product_id , username  = row
    sql_q = "select title,developer,specs,tags,release_date from products where id="+str(product_id)
    c.execute(sql_q)
    row = c.fetchone()
    title, developer, specs, tags, release_date = row
    result = "Review Number : " + str(query)+"<br><br>"
    result += "Review"+"<br>"
    result += "User Name : " + username + "<br>"
    result += "Review Context"+"<br>"+r_text+"<br><br>"

    result += str("-"*10) +"<br>"
    result += "Title : " + str(title) +"<br>"
    result += "Developer : " + str(developer)+"<br>"
    result += "Specs : " + str(specs)+"<br>"
    result += "Tags : " + str(tags)+"<br>"
    result += "Release Date : " + str(release_date)+"<br>"
    conn2 = sqlite3.connect("game_data_modi.db")
    sql_q = """select product_id , text from reviews where reviews.hours > 150 and length(text) > 100 and reviews.found_funny > 3 not like "0" AND product_id not like "%None%" """
    c = conn2.cursor()
    c.execute(sql_q)
    rows = c.fetchall()
    product_ids = []
    texts = []
    for product_id , text in rows:
        product_ids.append(product_id)
        texts.append(text)

    X = app.count_vect.transform(texts)
    X_tfidf = app.tfidf_transformer.transform(X)

    r_X = app.count_vect.transform([r_text])
    r_X_tfidf = app.tfidf_transformer.transform(r_X)

    sim = Similarity()
    result_siml = sim.calculate(X_tfidf,r_X_tfidf)
    ordered = sorted(range(len(result_siml)), key=lambda k: result_siml[k], reverse=True)
    products_list = []
    counter = 0
    for i in ordered:
        if product_ids[i] not in products_list and counter < 10:
            products_list.append(product_ids[i]) # list of products.
            counter += 1

    t_rows = []
    for pro in products_list:
        c.execute("select title,developer,genres,tags from products WHERE id=" + str(pro))
        rows = c.fetchall()
        for title, developer, tags, genres in rows:
            t_rows.append(r_Item(title, developer, tags, genres))

    table = RecommendedItemTable(t_rows)
    html = "<br>"+"----------Recommended----------"+"<br>"+str(table.__html__())
    html = html.replace("&#34;", "\"")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")

    return result + " <br> "+html




class ItemTable(Table):
    title = Col("Title")
    developer = Col("Developer")
    genres = Col("Genres")
    tags = Col("Tags")
    links = Col("reviews")

class RecommendedItemTable(Table):
    title = Col("Title")
    developer = Col("Developer")
    genres = Col("Genres")
    tags = Col("Tags")

class Item(object):
    def __init__(self, title,developer,genres,tags,reviews):
        self.title = title
        self.developer = developer
        self.tags = tags
        self.genres = genres
        self.links = reviews

class r_Item(object):
    def __init__(self, title,developer,genres,tags):
        self.title = title
        self.developer = developer
        self.tags = tags
        self.genres = genres

if __name__ == '__main__':
    if not hasattr(app,"count_vect"):
        with open('MyCountVectorizer.pickle', 'rb') as handle:
            app.count_vect = pickle.load(handle)
        with open('MyTfIdfVectorizer.pickle', 'rb') as f:
            app.tfidf_transformer = pickle.load(f)
        with open('MyNaiveBayes.pickle', 'rb') as NB:
            app.clf = pickle.load(NB)
    app.run(host='0.0.0.0', debug=True)

