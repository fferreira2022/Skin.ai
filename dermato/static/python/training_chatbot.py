import random
import numpy as np
import pickle
import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import WordNetLemmatizer

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\frede\\OneDrive\\Bureau\\skin_app\\dermato\\static\\json\\intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '{','}']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))
print(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

X_train = list(training[:, 0])
y_train = list(training[:, 1])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, validation_split=0.2, verbose=1)

model.save('static/h5/chatbot_model.h5', history)
print("Done")