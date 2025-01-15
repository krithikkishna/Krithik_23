from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Added this import
import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from flask import Flask, request, render_template, jsonify
import Preprocess
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction code here
    # For example, if you want to return a JSON response
    prediction_result = {'prediction': 'FAKE'}  # Replace this with your actual prediction result
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)




def process(input):
    data = pd.read_csv('dataset/data.csv')
    X = data.Text
    y = data.Label

    stemmer = WordNetLemmatizer()
    ''' cleaning, removing stop words'''
    # for text
    document_X = []
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        document_X.append(document)
    # for label
    document_Y = []
    for sen in range(0, len(y)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(y[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        document_Y.append(document)

    # splitting the data based on 70-30 ratio
    # X_train=document_X

    # y_train=document_Y

    X_train, X_test, y_train, y_test = train_test_split(document_X, document_Y, test_size=0.2, random_state=0)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)
    # Learn the vocabulary dictionary and return term-document matrix.

    # Input = test data article
    # test=["A week before Michael T. Flynn resigned "]

    test = [input]
    print(test)
    # encode document
    vector = count_vectorizer.transform(test)

    # To load the model one by one
    with open('dataset/DecisionTree', 'rb') as training_model:
        model = pickle.load(training_model)

    dpred = model.predict(vector.toarray())
    print(dpred)

    # To load the model one by one
    with open('dataset/RandomForest', 'rb') as training_model:
        model = pickle.load(training_model)

    rpred = model.predict(vector.toarray())
    print(rpred)

    # Predictions form all the above algorithms are either 1 or 0
    # where 1 denotes the real value and 0 denotes the fake value
    heva = int(dpred) + int(rpred)
    print(heva)
    news = ""
    if heva >= 1:
        news = 'Real'
    else:
        news = 'Fake'
    print(news)
    return news
