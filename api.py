import sqlite3
import gensim
import pickle
from gensim.parsing import *
from gensim.similarities.docsim import Similarity
from flask import Flask , render_template , request , url_for
from flask_restful import Resource, Api
from flask_table import Table, Col , LinkCol
import copy
import time
app = Flask(__name__)


def preprocess(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,stem_text,remove_stopwords,strip_multiple_whitespaces,strip_short]
    return gensim.parsing.preprocess_string(text,CUSTOM_FILTERS)


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
    query_arr = preprocess(query)
    dictionary = copy.deepcopy(app.dictionary)
    dictionary.add_documents([query_arr])
    bow_q = dictionary.doc2bow(query_arr)
    results = set()

    try:
        results = app.data[query_arr[0]]
    except KeyError as e:
        print("no key")
    for i in query_arr[1:]:
        try:
            results = results.intersection(app.data[i])
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
        pre = preprocess(review)
        bow = dictionary.doc2bow(pre)
        bows.append(bow)
        indices.append(idx)

    siml = Similarity(None, bows, num_features=len(dictionary))
    result_siml = siml[bow_q]

    ordered = sorted(range(len(result_siml)), key=lambda k: result_siml[k])
    r_list = []
    for i in ordered:
        r_list.append(indices[i])  # list of reviews.

    # sql_query = "select product_id from reviews where r_id IN "
    # sql_query += str(tuple(r_list))
    # print(sql_query)
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()
    # c.execute(sql_query)
    # rows = c.fetchall()
    sql_query = "select product_id from reviews where r_id = "
    d_pr = {}
    product_set = set()
    for i in r_list:
        c.execute("select product_id from reviews where r_id = " + str(i))
        rows = c.fetchall()
        for r in rows:
            print(r)
            if r[0] is None or type(r[0]) == str:
                print("None")
                continue
            if r[0] not in d_pr:
                d_pr[r[0]] = [i]
            else:
                d_pr[r[0]].append(i)
            product_set.add(r[0])
    t_rows = []
    for pro in product_set:
        print("select title,developer,genres,tags from products WHERE id=" + str(pro))
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
    return html

@app.route("/review",methods=['GET','POST'])
def show_review():
    if request.method == "GET":
        query = request.args.get('r_id')
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()
    sql_q = "select text , product_id , username from reviews where r_id="+str(query)
    c.execute(sql_q)
    row = c.fetchone()
    text , product_id , username  = row
    sql_q = "select title,developer,specs,tags,release_date from products where id="+str(product_id)
    c.execute(sql_q)
    row = c.fetchone()
    title, developer, specs, tags, release_date = row
    result = "Review Number : " + str(query)+"<br><br>"
    result += "Review"+"<br>"
    result += "User Name : " + username + "<br>"
    result += "Review Context"+"<br>"+text+"<br><br>"

    result += str("-"*10) +"<br>"
    result += "Title : " + str(title) +"<br>"
    result += "Developer : " + str(developer)+"<br>"
    result += "Specs : " + str(specs)+"<br>"
    result += "Tags : " + str(tags)+"<br>"
    result += "Release Date : " + str(release_date)+"<br>"
    return result




class ItemTable(Table):
    title = Col("Title")
    developer = Col("Developer")
    genres = Col("Genres")
    tags = Col("Tags")
    links = Col("reviews")

class Item(object):
    def __init__(self, title,developer,genres,tags,reviews):
        self.title = title
        self.developer = developer
        self.tags = tags
        self.genres = genres
        self.links = reviews

if __name__ == '__main__':

    app.dictionary = gensim.corpora.Dictionary.load("./dictionary.gensim")
    app.data = None
    print(app.dictionary)
    with open('dictionary.pickle', 'rb') as handle:
        app.data = pickle.load(handle)
    app.run(host='0.0.0.0',debug=True)
