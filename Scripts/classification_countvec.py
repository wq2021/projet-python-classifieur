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

# II. Extraction des features avec CountVectorizer

vectorizer = CountVectorizer()
X_train_CountVec = vectorizer.fit_transform(X_train)
words = vectorizer.get_feature_names()
print("*"*15 + "Traitement des features avec CountVectorizer" + "*"*15)

# 1. naive bayes 
text_cv_clf = Pipeline([('vect',CountVectorizer()),('clf',MultinomialNB())])
text_cv_clf.fit(X_train, y_train)

predicted_cv = text_cv_clf.predict(X_test)
print("Naive Bayes : MultinomialNB()")
print(classification_report(predicted_cv, y_test))

# 2. Logistic Regression
text_cv_clf_lr = Pipeline([('vect',CountVectorizer()),('clf',LogisticRegression())])
text_cv_clf_lr.fit(X_train, y_train)

predicted_cv_lr = text_cv_clf_lr.predict(X_test)
print("Logistic Regression : LogisticRegression()")
print(classification_report(predicted_cv_lr, y_test))

# 3. SVM 
text_cv_clf_svm = Pipeline([('vect',CountVectorizer()),('clf',SGDClassifier(loss="hinge", penalty="l2"))])
text_cv_clf_svm.fit(X_train, y_train)

predicted_cv_svm = text_cv_clf_svm.predict(X_test)
print("SVM : SGDClassifier()")
print(classification_report(predicted_cv_svm, y_test))

# 4. Decision Tree
text_cv_clf_dt = Pipeline([('vect',CountVectorizer()),('clf',DecisionTreeClassifier())])
text_cv_clf_dt.fit(X_train, y_train)

predicted_cv_dt = text_cv_clf_dt.predict(X_test)
print("Decision Tree : DecisionTreeClassifier()")
print(classification_report(predicted_cv_dt, y_test))