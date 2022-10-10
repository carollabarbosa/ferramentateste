import numpy as np
import json
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import joblib

import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
import nltk
nltk.download('punkt')

app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model2.pkl', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')
  
@app.route('/text')
def text():
   text= pd.read_csv('preprocessed05.csv')
   text_fake = 'text_falso'['label'].sum()
   text_true = 'text_verdadeiro'['label'].sum()
   resposta = {'text_verdadeiro': text_true}
   resposta = {'text_falso': text_fake}
   return jsonify (resposta)

@app.route('/predict', methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary()
    
    pred = model.predict([news])
    return render_template('index.html', prediction_text= 'Essa noticia é"{}"'.format(pred[0]))

if __name__ == "__main__":
    app.run()