import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from nltk.stem import WordNetLemmatizer
import csv



def process():
    i = 0
    data = []
    with open("dataset/train.tsv", encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            print(row[1], row[2])
            if "true" in row[1]:
                i = i + 1
                print("true")
                data.append([row[2], "REAL", 1])
            if "false" in row[1]:
                i = i + 1
                data.append([row[2], "FAKE", 0])
                print("false")

    with open("dataset/valid.tsv", encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            print(row[1], row[2])
            if "true" in row[1]:
                i = i + 1
                print("true")
                data.append([row[2], "REAL", 1])
            if "false" in row[1]:
                i = i + 1
                data.append([row[2], "FAKE", 0])
                print("false")

    with open("dataset/test.tsv", encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            print(row[1], row[2])
            if "true" in row[1]:
                i = i + 1
                print("true")
                data.append([row[2], "REAL", 1])
            if "false" in row[1]:
                i = i + 1
                data.append([row[2], "FAKE", 0])
                print("false")

    print("i value", i)
    print(data)

    with open("dataset/data.csv", 'w', encoding="utf8", newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Text", "Value", "Label"])
        for m in data:
            wr.writerow(m)

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
        # Initialize the `CountVectorizer`
        count_vectorizer = CountVectorizer(stop_words='english')
        # Get feature names using get_feature_names method
        feature_names = count_vectorizer.get_feature_names()

        # Now you can use the feature names
        print(feature_names)

        # Fit and transform the training data
        count_train = count_vectorizer.fit_transform(
            X_train)  # Learn the vocabulary dictionary and return term-document matrix.
        count_test = count_vectorizer.transform(X_test)

        # Get feature names using get_feature_names_out method
        feature_names = count_vectorizer.get_feature_names_out()

        # Now you can use the feature names
        print(feature_names)

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
    X_train, X_test, y_train, y_test = train_test_split(document_X, document_Y, test_size=0.2, random_state=0)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(
        X_train)  # Learn the vocabulary dictionary and return term-document matrix.
    count_test = count_vectorizer.transform(X_test)

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names())

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
    print(count_vectorizer.vocabulary_)

    # save BoW as a pickle object in Python
    with open('dataset/bow', 'wb') as picklefile:
        pickle.dump(count_vectorizer.get_feature_names(), picklefile)

