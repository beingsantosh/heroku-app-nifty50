I recently worked on Stock price prediction problem. It was about training the model with the last 10 years news headlines and stock prices. Neural network once trained, are ready for predicting the stock. It used beautifulsoup and selenium for web scrapping, later data was processed with the help of pandas,regex, spacy etc. Modelling was done through LSTM with GloVe embedding. In end, web application was deployed on heroku platform with routing on flask.


NIFTY50 stock prediction (NLP):
-	Designed the web application to predict the stock variation after processing the news headline (News → stock prediction)
-	Model: Bidirectional LSTMs with GloVe embeddings at input layer combination.
-	Data: Scrapped last 10 years news headlines from economictimes.com. It required Selenium and BeautifulSoup capability in blending to handle runtime Javascript generated data.
-	Data preprocessing: Pandas, regular expression(regex), spacy, numpy and statistics capabilities utilised to achieve clean data before training.
-	Web application: Implemented flask, HTML, CSS, bootstrap4, postman (API test).
-	Link: https://nifty50-prediction-app.herokuapp.com/
