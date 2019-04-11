import sqlite3
import json
import re


# cols = ["id","title","app_name","metascore","specs","developer","tags","url","genres","release_date","publisher","price","early_access","reviews_url","discount_price","n_reviews","sentiment"]
cols = ["product_id","user_id","recommended","hours","text","found_funny","compensation","page_order","products","early_access","username","page","date"]

def insert(table_name,data,conn):
    c = conn.cursor()
    head = "INSERT INTO " + table_name +" ("
    for i in range(0,len(cols)):
        head = head+ "\"" +cols[i]+"\""
        if i != len(cols)-1:
            head += ","
    head += ")"
    values = "VALUES ("
    index = 0

    for i in cols:
        if "product_id" not in data and "user_id" not in data and "hours" not in data:
            print(data)
            return
        if i in data:
            if type(data[i]) != str and type(data[i]) != int and type(data[i]) != float and type(data[i]) != bool :
                text = ""
                for t in data[i]:
                    if '\"' in t:
                        t = t.replace("\"","\'")
                    text += t+","
                values += "\"" + text + "\""
            elif i is "product_id" or i is "user_id" or i is"hours":
                values += str(data[i])
            elif i == "text":
                string = data[i]
                word1 = " ".join(re.findall("[a-zA-Z]+", string))
                values +="\""+ word1+"\""
            else:
                if '\"' in str(data[i]):
                    data[i] = data[i].replace("\"", "\'")
                values += "\"" + str(data[i]) + "\""
        else:
            values += "\"None\""
        if index < len(cols)-1:
            values +=","
        index += 1
    values += ")"

    c.execute(head +"\n"+ values)
    # print(head +"\n"+ values)
    c.close()
    conn.commit()
    return head +"\n"+ values


if __name__ == "__main__":
    conn = sqlite3.connect('game_data copy 3.db')
    with open("rep.txt") as f:
        try:
            for line in f:
                if len(line) < 10:
                    print("error : " , line)
                j_data = json.loads(line)
                insert("reviews", j_data,conn)
        except json.decoder.JSONDecodeError as jde:
            print("--------ERROR-------")
            print(jde)
            print(line)
            print("--------ERROR-------")

    conn.close()