import os
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from pre_traitement import load_datasets
from pre_traitement import importer_data_csv
from pre_traitement import echantillonner


# X_train, y_train, X_test, y_test = load_datasets()


class TesteurClf:
    def __init__(self, vect, clf, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

        if vect == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.vec = "tfidf"
        elif vect == "countV":
            self.vectorizer = CountVectorizer()
            self.vec = "countVectorizer"
        
        if clf == "bayes":
            self.classifieur = MultinomialNB()
            self.clf = "Naive Bayes : MultinomialNB()"
        elif clf == "LR":
            self.classifieur = LogisticRegression()
            self.clf = "Logistic Regression : LogisticRegression()"
        elif clf == "SVM":
            self.classifieur = SGDClassifier()
            self.clf = "SVM : SGDClassifier()"
        elif clf == "tree":
            self.classifieur = DecisionTreeClassifier()
            self.clf = "Decision Tree : DecisionTreeClassifier()"
    
    def modeliser(self):
        text_clf = Pipeline([('vect', self.vectorizer),('clf', self.classifieur)])
        text_clf.fit(self.X_train, self.y_train)
        predicted = text_clf.predict(self.X_test)
        report = classification_report(predicted, self.y_test)

        return self.clf, self.vec, report







def main():
    parser = argparse.ArgumentParser(description="créer des répertoires de data bien formés et normaliser le texte")
    parser.add_argument("taille", type=int, help="taille de l'échantillon")
    parser.add_argument("testsize", type=float, help="le pourcentage de sous-partie test de l'échantillon")
    parser.add_argument("--Stop", type=bool, help="booléen, choisir si on traite les stopwords ou pas")
    parser.add_argument("--ficCsv", help="nom du fichier corpus csv")
    parser.add_argument("--lowercase", type=bool, help="donner le True si vous voulez mettre tous les mots en minuscules")
    
    args = parser.parse_args()
    
    size = args.taille
    test_size = args.testsize
    stopword = args.Stop
    lowercase = args.lowercase
    fic_csv = "french_tweets.csv"
    if args.ficCsv:
        fic_csv = args.ficCsv
        

    nom_csv_sample = echantillonner(size, ficName=fic_csv)
    X_train, y_train, X_test, y_test = importer_data_csv(nom_csv_sample, test_size, lowercase, stopword)

    lst_vect = ["tfidf", "countV"]
    lst_clf = ["bayes", "LR", "SVM", "tree"]
    for vec in lst_vect:
        for clf in lst_clf:
            testeur = TesteurClf(vec, clf, X_train, y_train, X_test, y_test)
            entete, vectorizer, report = testeur.modeliser()
            print(entete, vectorizer)
            print(report)

if __name__ == "__main__":
    main()