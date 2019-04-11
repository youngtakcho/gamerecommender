Requirements for this project
- Python 3.6 or above
- Flask and flask-table
- gensim
- sklearn
- Scraper
- Steam scraper 

Why we need flask?
- This project is web application with a server-client model. Web application server communicates with clients under http protocol and flask provides easy way to make web application server.

Why we need gensim?
- Gensim is a good library for natural language processing. But for this project, only preprocessing parts of gensim are used.

How to install flask?
- pip install flask flask-table

How to install gensim?
- pip install gensim

Additional info :
Whatever your OS is, there is firewall for protecting your system even if you use hosting server from a provider. For servicing your application, I recommend to add rule for web service to the firewall. Flask uses the port number 5000 for web service. Add 5000 to your inbound rule to your firewall.

How to Do step.
 1.	collect data by using Steam Scraper
 2.	change collected review file’s name to rep.txt and run Data_to_DB.py
 3.	run i_index.py to build inverted index dictionary file.
 4. select a genre from genres column run select_genre.py
 5. make tf-idf and gensim similarity model run make_model.py and make_gensim_dic.py 
 6.	run api.py to run http server.
