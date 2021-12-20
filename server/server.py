#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load tempfile for temporary dir creation
import sys, os, tempfile
# load misc utils
import json, random
# import uuid
from werkzeug.utils import secure_filename
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# load libraries for string proccessing
import re, string

# load libraries for API proccessing
from flask import Flask, jsonify, flash, request, Response, redirect, url_for, abort, render_template, send_from_directory

# A Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.
from flask_cors import CORS

import gensim

from vec2graph import visualize

import pathlib
pathlib.Path('./output/').mkdir(parents=True, exist_ok=True) 

__author__ = "Kyrylo Malakhov <malakhovks@nas.gov.ua>"
__copyright__ = "Copyright (C) 2020 Kyrylo Malakhov <malakhovks@nas.gov.ua>"

app = Flask(__name__)
CORS(app)

"""
Limited the maximum allowed payload to 16 megabytes.
If a larger file is transmitted, Flask will raise an RequestEntityTooLarge exception.
"""
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

"""
Set the secret key to some random bytes. Keep this really secret!
How to generate good secret keys.
A secret key should be as random as possible. Your operating system has ways to generate pretty random data based on a cryptographic random generator. Use the following command to quickly generate a value for Flask.secret_key (or SECRET_KEY):
$ python -c 'import os; print(os.urandom(16))'
b'_5#y2L"F4Q8z\n\xec]/'
"""
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.secret_key = os.urandom(42)

config_flag = 'ua'

# * Load models from config file to memory
# ! Caution: Loading a large number of models requires a significant amount of RAM
try:
    with open('./config.models.simple.ua.json') as config_file:
        models = json.load(config_file)
except IOError as e:
    logging.error(e, exc_info=True)
if 'word2vec' not in models["models"]:
    raise ValueError("No word2vec models in given config file (config.models.simple.ua.json)")
else:
    models_array = []
    models_word2vec = models["models"]["word2vec"]
    for model_index, model in enumerate(models_word2vec):
        # Load and init word2vec model
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model['link'])
        word_vectors.init_sims(replace=True)
        models_array.append(word_vectors)
    del word_vectors

# * Load models-en from config file to memory
try:
    with open('./config.models.simple.en.json') as config_file_en:
        models_en = json.load(config_file_en)
except IOError as e:
    logging.error(e, exc_info=True)

"""
from gensim.models import Word2Vec as WV_model
model = WV_model.load('./models/suhomlinskyy.lowercased.lemmatized.word2vec.500d')
* For word2vec2tensor, the model should be in "word2vec_format" (this isn't same as result of .save())
* You need to call model.wv.save_word2vec_format(...), and after this, use word2vec2tensor on result file.
model.wv.save_word2vec_format('suhomlinskyy.lowercased.lemmatized.word2vec.500d')
# * switch to the KeyedVectors instance
word_vectors_suhomlinskyy = model.wv
word_vectors_suhomlinskyy.init_sims(replace=True)
del model
"""

def getExistsWordsInModel(words, keyed_vectors):
    exists = []
    for word in words:
        if word in keyed_vectors.vocab:
            exists.append(word)
    return exists

# * models list
@app.route('/api/models')
def get_models_list():
    return jsonify(models)

# * computational endpoints
@app.route('/api/word2vec/random/words', methods=['GET'])
def get_random_words():
    try:
        random_words = random.sample(models_array[request.args.get('model', type = int)].index_to_key, request.args.get('number', type = int))
        return jsonify({"random": random_words})
    except Exception as e:
        logging.error(e, exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/word2vec/similar', methods=['POST'])
def get_similar():
    if not request.json or not 'word' in request.json or not 'words' in request.json:
        abort(400)
    try:
        if request.args.get('model', type = int):
            most_similar_word = models_array[request.args.get('model', type = int)].most_similar_to_given(request.json['word'], request.json['words'])
        else:
            most_similar_word = models_array[0].most_similar_to_given(request.json['word'], request.json['words'])
        return jsonify({"similar": most_similar_word})
    except KeyError:
        return jsonify({"error": {"KeyError": "word does not exist in the word2vec model" , "word": request.json['word']}}), 400

# * vec2graph
@app.route('/api/vec2graph')
def get_vec2graph_viz():
    try:
        visualize('./output/', models_array[request.args.get('model', type = int)], request.args.get('word', type = str), depth=0, topn=100, threshold=request.args.get('threshold', type = int), edge=1, sep=False, library="web")
        return render_template('index.html')
    except Exception as e:
        logging.error(e, exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # default port = 5000
    app.run(host = '0.0.0.0')
    # app.run(host = '0.0.0.0', port=3000)