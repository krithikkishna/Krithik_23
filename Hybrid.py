from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import Predict

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score




def process():
    # group / ensemble of models
    estimator = []
    estimator.append(('DTC', DecisionTreeClassifier()))
    estimator.append(('RF', RandomForestClassifier(n_estimators=1000, random_state=0)))

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

    # Voting Classifier with hard voting
    vot_hard = VotingClassifier(estimators=estimator, voting='hard')
    vot_hard.fit(count_train, y_train)
    y_pred = vot_hard.predict(count_test)

    # using accuracy_score metric to predict accuracy
    score = accuracy_score(y_test, y_pred)
    print("Hard Voting Score ", score)

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
    print("MSE VALUE FOR Hybrid IS %f " % mse)
    print("MAE VALUE FOR Hybrid IS %f " % mae)
    print("R-SQUARED VALUE FOR Hybrid IS %f " % r2)
    rms = np.sqrt(mean_squared_error(y_test1, y_pred1))
    print("RMSE VALUE FOR Hybrid IS %f " % rms)
    ac = accuracy_score(y_test1, y_pred1)
    print("ACCURACY VALUE Hybrid IS %f" % ac)
    print("---------------------------------------------------------")

    result2 = open('results/HYMetrics.csv', 'w')
    result2.write("Parameter,Value" + "\n")
    result2.write("MSE" + "," + str(mse) + "\n")
    result2.write("MAE" + "," + str(mae) + "\n")
    result2.write("R-SQUARED" + "," + str(r2) + "\n")
    result2.write("RMSE" + "," + str(rms) + "\n")
    result2.write("ACCURACY" + "," + str(ac) + "\n")
    result2.close()

    df = pd.read_csv('results/HYMetrics.csv')
    acc = df["Value"]
    alc = df["Parameter"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    explode = (0.1, 0, 0, 0, 0)

    fig = plt.figure()
    plt.bar(alc, acc, color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title(' Hybrid Metrics Value')
    plt.savefig('results/HYMetricsValue.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()
