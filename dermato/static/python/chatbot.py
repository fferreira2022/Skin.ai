import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\frede\\OneDrive\\Bureau\\skin_app\\dermato\\static\\json\\intents.json').read())

words = pickle.load(open('C:\\Users\\frede\\OneDrive\\Bureau\\skin_app\\dermato\\static\\pickle\\words.pkl', 'rb'))
classes = pickle.load(open('C:\\Users\\frede\\OneDrive\\Bureau\\skin_app\\dermato\\static\\pickle\\classes.pkl', 'rb'))
model = load_model('C:\\Users\\frede\\OneDrive\\Bureau\\skin_app\\dermato\\static\\h5\\chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Bot is running...')

while True:
    message = input('You: ')
    ints = predict_class(message)
    response = get_response(ints, intents)
    print(response)
    
        