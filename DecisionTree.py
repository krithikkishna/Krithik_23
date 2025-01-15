import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import Hybrid
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def process():
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
    X_train, X_test, y_train, y_test = train_test_split(document_X, document_Y, test_size=0.2, random_state=0)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(
        X_train)  # Learn the vocabulary dictionary and return term-document matrix.
    count_test = count_vectorizer.transform(X_test)

    classifier = DecisionTreeClassifier()
    classifier.fit(count_train, y_train)

    y_pred = classifier.predict(count_test)

    print(accuracy_score(y_test, y_pred))

    a = confusion_matrix(y_test, y_pred)
    a1 = a.flatten()
    print(a1)
    print(classification_report(y_test, y_pred))

    labels = 'TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative'
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    # Plot
    plt.pie(a1, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.savefig('results/DTConfusionMatrix.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()

    # save our model as a pickle object in Python
    with open('dataset/DecisionTree', 'wb') as picklefile:
        pickle.dump(classifier, picklefile)

    result2 = open("results/resultDT.csv", "w")
    result2.write("ID,Predicted Value" + "\n")
    for j in range(len(y_pred)):
        result2.write(str(j + 1) + "," + str(y_pred[j]) + "\n")
    result2.close()

    print(len(y_test))
    print(len(y_pred))

    y_test1 = []
    for item in y_test:
        y_test1.append(float(item))

    y_pred1 = []
    for item in y_pred:
        y_pred1.append(float(item))

    mse = mean_squared_error(y_test1, y_pred1)
    mae = mean_absolute_error(y_test1, y_pred1)
    r2 = r2_score(y_test1, y_pred1)

    print("---------------------------------------------------------")
    print("MSE VALUE FOR Decision Tree IS %f " % mse)
    print("MAE VALUE FOR Decision Tree IS %f " % mae)
    print("R-SQUARED VALUE FOR Decision Tree IS %f " % r2)
    rms = np.sqrt(mean_squared_error(y_test1, y_pred1))
    print("RMSE VALUE FOR Decision Tree IS %f " % rms)
    ac = accuracy_score(y_test1, y_pred1)
    print("ACCURACY VALUE Decision Tree IS %f" % ac)
    print("---------------------------------------------------------")

    result2 = open('results/DTMetrics.csv', 'w')
    result2.write("Parameter,Value" + "\n")
    result2.write("MSE" + "," + str(mse) + "\n")
    result2.write("MAE" + "," + str(mae) + "\n")
    result2.write("R-SQUARED" + "," + str(r2) + "\n")
    result2.write("RMSE" + "," + str(rms) + "\n")
    result2.write("ACCURACY" + "," + str(ac) + "\n")
    result2.close()

    df = pd.read_csv('results/DTMetrics.csv')
    acc = df["Value"]
    alc = df["Parameter"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    explode = (0.1, 0, 0, 0, 0)

    fig = plt.figure()
    plt.bar(alc, acc, color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title(' Decision Tree Metrics Value')
    plt.savefig('results/DTMetricsValue.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()

