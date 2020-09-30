from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    print('hello')
    condition = int_features[0] #condition = bedroom
    yr_built = int_features[4]
    int_features.append(condition);
    print('int features =',int_features)
    if yr_built < 1900:
        yr_built = 1900
    if yr_built >2015:
        yr_built = 2015
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price of House = ${}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
