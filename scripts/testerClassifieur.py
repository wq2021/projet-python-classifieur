# -*- coding: utf-8 -*-
# pour exécuter le script:
# python testerClassifieur.py    taille_echantillon    pourcentage_test    --Stop True --lowercase True
# exemple:
# python testerClassifieur.py 500 0.3 --Stop True --lowercase True

"""
Module pour tester tous les classifieurs
"""

import os
import re
import numpy as np
import argparse

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from pre_traitement import load_datasets
from pre_traitement import importer_data_csv
from pre_traitement import echantillonner


class TesteurClf:
    """
    Cette classe va instancier le vectorizeur, le classifieur selon les infos fournis quand être instancié
    et puis entraîner un modèle et renvoyer son résultat quand lancé la méthode    modeliser() 
    """
    def __init__(self, vect, clf, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

        if vect == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.vec = "tfidf"
        elif vect == "countV":
            self.vectorizer = CountVectorizer()
            self.vec = "countVectorizer"
        
        if clf == "bayes":
            self.classifieur = MultinomialNB(alpha=1)
            self.clf = "Naive Bayes : MultinomialNB()"
        elif clf == "LR":
            self.classifieur = LogisticRegression(solver='liblinear')
            self.clf = "Logistic Regression : LogisticRegression()"
        elif clf == "SVM":
            self.classifieur = SGDClassifier(loss='log', penalty='elasticnet')
            self.clf = "Linear classifiers with SGD training : SGDClassifier()"
        elif clf == "tree":
            self.classifieur = DecisionTreeClassifier(max_depth=6)
            self.clf = "Decision Tree : DecisionTreeClassifier()"
        elif clf == "SVC":
            self.classifieur = SVC(C=4, gamma=2, kernel='rbf')
            self.clf = "C-Support Vector Classification : SVC()"
    
    def modeliser(self):
        """
            construire le pipeline de traitement et puis entrainer et tester 
            et renvoyer le nom de classifieur, le genre de vectorizer et le rapport de test
        """
        text_clf = Pipeline([('vect', self.vectorizer),('clf', self.classifieur)])
        text_clf.fit(self.X_train, self.y_train)
        predicted = text_clf.predict(self.X_test)
        report = classification_report(predicted, self.y_test)
        return self.clf, self.vec, report


def trouve_meilleurs_para(X_train, y_train):
    """
        fonction sert à tester dans un plus petit échantillon pour obtenir les paramètres optimisés
    """
    # vectorisation
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    #instancier les classifieurs et donner les paras à optimiser
    clf_DT = DecisionTreeClassifier()
    param_grid_DT = {'max_depth': [1,2,3,4,5,6]}

    clf_MNB = MultinomialNB()
    param_grid_MNB = {'alpha': [0.01,0.05,0.1,0.25,0.5,0.7,1]}

    clf_Logit = LogisticRegression()
    param_grid_logit = {'solver': ['liblinear','lbfgs','newton-cg','sag']}

    clf_svc = SVC()
    param_grid_svc = {'kernel':('linear', 'rbf'), 
                    'C':[1, 2, 4], 
                    'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}

    clf_sgd = SGDClassifier()
    param_grid_sgd = {'loss' : ['hinge','modified_huber', 'log'],
                    'penalty': ['l1', 'l2', 'elasticnet']}

    # faire le test
    clf = [clf_DT, clf_MNB, clf_Logit, clf_svc, clf_sgd]
    param_grid = [param_grid_DT, param_grid_MNB, param_grid_logit, param_grid_svc, param_grid_sgd]
    for i in range(0,5):
        grid=GridSearchCV(clf[i], param_grid[i], scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        print (grid.best_params_,': ',grid.best_score_)


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
    

    #ancienne solution
    # X_train, y_train, X_test, y_test = load_datasets()

    nom_csv_sample = echantillonner(size, ficName=fic_csv)
    X_train, y_train, X_test, y_test = importer_data_csv(nom_csv_sample, test_size, lowercase, stopword)

    

    # choisir l'un de ces deux blocs de code pendant le lancement
    
    # 1) code pour obtenir les paramètres optimisés de chaque classifieur 
    # trouve_meilleurs_para(X_train_tfidf, y_train)

    # 2) code pour obtenir le rapport
    lst_vect = ["tfidf", "countV"]
    lst_clf = ["bayes", "LR", "SVM", "tree", "SVC"]
    for vec in lst_vect:
        for clf in lst_clf:
            testeur = TesteurClf(vec, clf, X_train, y_train, X_test, y_test)
            entete, vectorizer, rapport = testeur.modeliser()
            print(entete, vectorizer)
            print(rapport)

if __name__ == "__main__":
    main()