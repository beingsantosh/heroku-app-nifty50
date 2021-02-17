from flask import Flask, request, render_template, url_for
import numpy as np
from tensorflow.keras.models import load_model
from predictions.util_prediction import news_to_int, padding_news, predict

app = Flask(__name__)

"""
Below code belongs to prediction 
"""
vocab_filepath = 'vocab_to_int.json'
max_daily_length = 50
max_headline_length = 10
embedding_dim = 100
news = "India plans to invest heavily in Energy sector."
load_model = load_model('./predictions/simple_nifty.h5')

# predicted_price = predict(news, load_model, vocab_filepath, max_daily_length)

# print('Finish')


@app.route('/')
def index():
    return render_template('bootstrap.html')


@app.route('/predict', methods=['POST'])
def index2():
    if request.method == 'POST':
        headlines = request.form.get('name_headlines')
        prediction = predict(headlines, load_model, vocab_filepath, max_daily_length)
    return render_template('bootstrap.html', prediction=round(prediction,3))

# @app.route('/predict', methods=['GET'])
# def index2():
#     # print(request.atrs.get['name_headlines'])
#     return render_template('bootstrap_with_prediction.html',
#                            prediction=f'Expected variation in NIFTY50 stock exchange would be: {}')


# # check
# def predict_f():
#     if int(np.random.random() * 10) % 2 == 0:
#         num = - round(np.random.random() * 1000, 2)
#         return num
#     else:
#         num = round(np.random.random() * 1000, 2)
#         return num


if __name__ == '__main__':
    app.run(debug=False)
