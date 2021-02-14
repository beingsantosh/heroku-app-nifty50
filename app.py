from flask import Flask,request, render_template, url_for
#from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np

app = Flask(__name__)



# @app.route('/', methods = ['POST', 'GET'])
#
# def index():
#
#     if request.method == 'POST':
#         return 'hello'
#     else:
#         return render_template('index.html')


@app.route('/', methods = ["GET",'POST'])
def index1():
    prediction_value = predict_f()

    if request.method == ['POST']:
        global news
        news = request.form.get('name_headlines')
        if news is None:
            print('taken', news)
        else:
            print('no', news)
        return 'Hello'
    else:
        return render_template('bootstrap.html')

@app.route('/predict')
def index2():
    prediction_value = predict_f()
    # if news is None:
    #     print('taken1', news)
    # else:
    #     print('noa',news)
    return render_template('bootstrap_with_prediction.html', prediction= f'Expected variation in NIFTY50 stock exchange would be: {prediction_value}' )


# check
def predict_f():
    if int(np.random.random()*10)%2 ==0:
        num = - round(np.random.random()*1000,2)
        return (num)
    else:
        num = round(np.random.random()*1000,2)
        return (num)

# if __name__ == '__main__':
app.run(debug=False)