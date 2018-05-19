
import numpy as np
import time
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
# Copyright 2015 Google Inc. All Rights Reserved.
import logging

from flask import Flask,request,Response


app = Flask(__name__)

def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print (return_results)
    return return_results
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.route('/api/v1/ChatbotAmadeus', methods=['POST'])
def Amadeus():
    """Return a friendly HTTP greeting."""
    data = request.get_json(silent=True)
    try:
    	deff = data["sentence"]
    	prediccion = classify(deff)
    	if(len(prediccion)>=1):
    		data = {
	    		'intent'  : prediccion[0][0],
	    		'score' : prediccion[0][1]
    		}
    		js = json.dumps(data)
    		resp = Response(js, status=200, mimetype='application/json')
    		return resp
    	else:
    		data = {
	    		'intent'  : None
    		}
    		js = json.dumps(data)
    		resp = Response(js, status=200, mimetype='application/json')
    		return resp
    except ValueError:
    	data = {
    		'Message'  : "Request Incorrecto"
    	}
    	js = json.dumps(data)
    	resp = Response(js, status=400, mimetype='application/json')
    	return resp


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    # [END app]
    stemmer = LancasterStemmer()
    global words 
    global classes 
    global synapse 
    global synapse_0 
    global synapse_1 


    # probability threshold
    ERROR_THRESHOLD = 0.2
    # load our calculated synapse values
    synapse_file = 'synapses.json' 
    with open(synapse_file) as data_file: 
        synapse = json.load(data_file) 
        synapse_0 = np.asarray(synapse['synapse0']) 
        synapse_1 = np.asarray(synapse['synapse1'])
        words= np.asarray(synapse['words'])
        classes= np.asarray(synapse['classes']) 
    app.run(host='127.0.0.1', port=8080, debug=True)

# compute sigmoid nonlinearity


