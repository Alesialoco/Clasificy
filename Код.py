import requests
import html2text
from flashtext.keyword import KeywordProcessor
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import nltk
import datetime
import json
import time


synapse_0 = ""
synapse_1 = ""
Class_1_keywords = ['Office', 'School', 'phone', 'Technology', 'Electronics', 'Cell', 'Business', 'Education',
                    'Classroom']
Class_2_keywords = ['Restaurant', 'Hospitality', 'Tub', 'Drain', 'Pool', 'Filtration', 'Floor', 'Restroom', 'Consumer',
                    'Care', 'Bags', 'Disposables']
Class_3_keywords = ['Pull', 'Lifts', 'Pneumatic', 'Emergency', 'Finishing', 'Hydraulic', 'Lockout', 'Towers', 'Drywall',
                    'Tools', 'Packaging', 'Measure', 'Tag ']
keywords = Class_1_keywords + Class_2_keywords + Class_3_keywords

stemmer = LancasterStemmer()
training = []
output = []
train_data = []
words = []
classes = []
documents = []
ignore_words = ['?']
html_code = ""

print("Введите ссылку:")
url = input()
try:
    page = requests.get(url)
    html_code = page.content
except Exception as e:
    print(e)

h = html2text.HTML2Text()
h.ignore_links = True
try:
    text = h.handle(html_code)
    text_from_html = text.replace("\n", " ")
except Exception as e:
    print(e)

kp0 = KeywordProcessor()
kp1 = KeywordProcessor()
kp2 = KeywordProcessor()
kp3 = KeywordProcessor()
for word in keywords:
    kp0.add_keyword(word)
for word in Class_1_keywords:
    kp1.add_keyword(word)
for word in Class_2_keywords:
    kp2.add_keyword(word)
for word in Class_3_keywords:
    kp3.add_keyword(word)


def percentage1(dum0, dumx):
    try:
        ans = float(dumx) / float(dum0)
        ans = ans * 100
    except:
        return 0
    else:
        return ans


def find_class():
    category = ''
    x = str(text_from_html)
    y0 = len(kp0.extract_keywords(x))
    y1 = len(kp1.extract_keywords(x))
    y2 = len(kp2.extract_keywords(x))
    y3 = len(kp3.extract_keywords(x))
    per1 = float(percentage1(y0, y1))
    per2 = float(percentage1(y0, y2))
    per3 = float(percentage1(y0, y3))
    if y0 == 0:
        category = 'None'
    else:
        if per1 >= per2 and per1 >= per3:
            category = 'Офис, образование или технологии'
        elif per2 >= per3 and per2 >= per1:
            category = 'Потребительские товары'
        elif per3 >= per1 and per3 >= per2:
            category = 'Промышленность'
    return category


data = pd.read_csv('data.csv')
data = data[pd.notnull(data['tokenized_source'])]
data = data[data.Category != 'None']
for index, row in data.iterrows():
    train_data.append({"class": row["Category"], "sentence": row["text"]})
for pattern in train_data:
    w = nltk.word_tokenize(pattern['sentence'])
    words.extend(w)
    documents.append((w, pattern['class']))
    if pattern['class'] not in classes:
        classes.append(pattern['class'])
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    training.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)
print("# words", len(words))
print("# classes", len(classes))


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def train(X, y, hidden_neurons=10, alpha=1.0, epochs=50000, dropout=False, dropout_percent=0.5):
    global synapse_0, synapse_1
    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)
    last_mean_error = 1
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1
    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    for j in iter(range(epochs + 1)):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        if dropout:
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                    1.0 / (1 - dropout_percent))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        layer_2_error = y - layer_2
        if (j % 10000) == 0 and j > 5000:
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        if j > 0:
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
    now = datetime.datetime.now()
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"
    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("saved synapses to:", synapse_file)


def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    l0 = x
    l1 = sigmoid(np.dot(l0, synapse_0))
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


ERROR_THRESHOLD = 0.2
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])


def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print(f"\n classification: {return_results}s")
    return return_results


X = np.array(training)
y = np.array(output)
start_time = time.time()
train(X, y, hidden_neurons=10, alpha=0.1, epochs=50000, dropout=False, dropout_percent=0.2)
elapsed_time = time.time() - start_time
print("processing time:", elapsed_time, "seconds")

classify(
    "Switchboards Help KA36200 About Us JavaScript seems to be disabled in your browser You must have JavaScript "
    "enabled in your browser to utilize the functionality of this website Help Shopping Cart 0 00 You have no items in "
    "your shopping cart My Account My Wishlist My Cart My Quote Log In BD Electrical Worldwide Supply Remanufacturing "
    "the past SUSTAINING THE FUTURE Hours and Location Michigan Howell")
