import sqlite3

conn = sqlite3.connect("game_data_modi.db")

c = conn.cursor()
q = """select id,genres,tags from products where p_genre like "Free to Play" """


rows = c.execute(q)
u_c = conn.cursor()
for id,g,tags in rows:
    if tags == "None":
        continue
    tag_arr = tags.split(",")
    g_arr = g.split(",")
    t_set = set(tag_arr)
    t_set.union(set(g_arr))
    g = "Free to Play"
    if "Racing" in t_set:
        g = "Racing"
    elif "Sports" in t_set:
        g = "Sports"
    elif "Simulation" in t_set:
        g = "Simulation"
    elif "Adventure" in t_set:
        g ="Adventure"
    elif "Indie" in t_set:
        g ="Indie"
    elif "RPG" in t_set:
        g ="RPG"
    elif "Action" in t_set:
        g ="Action"
    elif "Strategy" in t_set:
        g ="Strategy"
    q = """update products set p_genre="%s" where id=%d"""%(g,id)
    u_c.execute(q)

conn.commit()
conn.close()