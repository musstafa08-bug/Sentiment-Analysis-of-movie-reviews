from flask import Flask, render_template, flash, request, url_for, redirect, session
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

image_folder = os.path.join('static', 'emojis')

app = Flask(__name__, template_folder='template')
app.config['upload_folder'] = image_folder

def init():
    global model, graph
    model = load_model('sentiment_analysis_model.h5')
   # graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', 'GET'])
def sentiment_analysis_pred():
    if request.method == 'POST':
        text = request.form['text']
        Sentiment = ''
        max_review_len = 700
        word_to_idx = imdb.get_word_index()
        special_chars = re.compile("[^A-Za-z0-9]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(special_chars, "", text.lower())

        words = text.split()
        x_test = [[word_to_idx[word] if (word in word_to_idx and word_to_idx[word] <=20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, max_len = max_review_len)
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            probability = model.predict(array([vector][0]))[0][0]
            class_1 = model.predict_classes(array([vector][0]))[0][0]

        if class_1 == 0:
            Sentiment = 'Negative'
            img_filename = os.path.join(app.config['upload_folder'], 'sad_emoji.png')

        else:
            Sentiment = 'Positive'
            img_filename = os.path.join(app.config['upload_folder'], 'smiling_emoji.png')

    return render_template('home.html', text = text, Sentiment = Sentiment, probability = probability, image = img_filename)

if __name__ == '__main__':
    init()
    app.run(debug=True)