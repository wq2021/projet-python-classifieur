import os
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from pre_traitement import load_datasets


X_train, y_train, X_test, y_test = load_datasets()

class TesteurClf(mode):
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = load_datasets()

        if mode == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif mode == "countV":
            self.vectorizer = CountVectorizer()
        vectorizer.fit_transform(self.X_train)



# # I. Extraction des features avec TF-IDF
# tfidf_vectorizer = TfidfVectorizer()
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# words = tfidf_vectorizer.get_feature_names()
# print("*"*15 + "Traitement des features avec TF-IDF" + "*"*15)

# # II. Extraction des features avec CountVectorizer
# vectorizer = CountVectorizer()
# X_train_CountVec = vectorizer.fit_transform(X_train)
# words = vectorizer.get_feature_names()
# print("*"*15 + "Traitement des features avec CountVectorizer" + "*"*15)


# 1. naive bayes 

# Pipeline
text_clf = Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
print("Naive Bayes : MultinomialNB()")
print(classification_report(predicted, y_test))

# 2. Logistic Regression
text_clf_lr = Pipeline([('vect',TfidfVectorizer()),('clf',LogisticRegression())])
text_clf_lr.fit(X_train, y_train)

predicted_lr = text_clf_lr.predict(X_test)
print("Logistic Regression : LogisticRegression()")
print(classification_report(predicted_lr, y_test))


# 3. SVM 
text_clf_svm = Pipeline([('vect',TfidfVectorizer()),('clf',SGDClassifier(loss="hinge", penalty="l2"))])
text_clf_svm.fit(X_train, y_train)

predicted_svm = text_clf_svm.predict(X_test)
print("SVM : SGDClassifier()")
print(classification_report(predicted_svm, y_test))

# 4. Decision Tree
text_clf_dt = Pipeline([('vect',CountVectorizer()),('clf',DecisionTreeClassifier())])
text_clf_dt.fit(X_train, y_train)

predicted_dt = text_clf_dt.predict(X_test)
print("Decision Tree : DecisionTreeClassifier()")
print(classification_report(predicted_dt, y_test))