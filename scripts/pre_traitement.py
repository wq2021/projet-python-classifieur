# -*- coding: utf-8 -*-
# pour exécuter le script:
# python pre_traitement.py 20 --Stop True --ficCsv echantillon_200.csv --lowercase True


"""
Module pour regrouper tous les fonctions qui manipuler les fichiers 
(échantillonnage, transformation du csv au répertoire/au datasets demandés par sklearn)
"""

import re 
import os
import sys
import numpy as np
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import normalisation as nor


# localiser le corpus, echantillon, repertoire créé
path = f"{sys.path[0]}/../ressources" 



def echantillonner(taille, ficName="french_tweets.csv"):
    """
        échantillonnage aléatoire du corpus original, fichier sorti garde l'organisation originale du corpus,
        renvoie le nom du fichier créé
        créer un nouveau csv dans le répertoire ressources, imprimer dans le terminal l'info de l'échantillon

        nom du fichier sorti:  echantillon_taille.csv      ex. echantillon_100.csv
        taille du corpus original (avec doublons):
            0(negatif) 771604
            1(positif) 755120
    """
    # fixer le répertoire du corpus et l'adresse du fichier sortie

    fichier = f"{path}/{ficName}"
    df = pd.read_csv(fichier)

    # supprimer les doublons
    df2 = df.drop_duplicates()
    df2 = df2.sample(n=taille)
    # imprimer l'info de l'echantillon dans terminal
    print("################\nInfo sur les catégories:")
    print(df2["label"].value_counts())
    print("################")

    df2.to_csv(f"{path}/echantillon_{taille}.csv", index=False)
    return f"echantillon_{taille}.csv"


def create_rep_split():
    """
        créer les répertoires de test et de train pour sauvegarder les fichiers txt plus tard
    """
    # tester et supprimer d'abord le test et le train existants
    if os.path.exists(f"{path}/test"):
        shutil.rmtree(f"{path}/test")
    if os.path.exists(f"{path}/train"):
        shutil.rmtree(f"{path}/train")
        
    for partie in ["test","train"]:
        for categorie in ["positif","negatif"]:
            os.makedirs(f"{path}/{partie}/{categorie}")


def corpus_separation(csvfic, minuscule, stopword):
    """
        mettres le contenu du fichier csv dans les répertoires, en respectant la structure demandée
        exemple train:
        -train
            -positif
                fic1.txt fic2.txt ...
            -negatif
                fic3.txt  fic4.txt ...
    """
    lst_stopwords = nor.importer_stopwords("stopwords_fr.txt")

    with open(f'{path}/{csvfic}','r',encoding="utf8") as entree:
        corpus_negatif = []
        corpus_positif = []

        # séparer les tweet selon leur catégorie
        for ligne in entree:
            if ligne.startswith('0'):
                phrase = ligne[2:]
                phrase = nor.nettoyage(phrase, lst_stopwords, lower=minuscule, stopword=stopword)
                corpus_negatif.append(phrase)
                
            if ligne.startswith('1'):
                phrase = ligne[2:]
                phrase = nor.nettoyage(phrase, lst_stopwords, lower=minuscule, stopword=stopword)             
                corpus_positif.append(phrase)
        
        # transformer en nparray, pour qu'il soit plus vite
        corpus_negatif = np.array(corpus_negatif)
        train_neg_taille = len(corpus_negatif)*0.8
        corpus_positif = np.array(corpus_positif)
        train_pos_taille = len(corpus_positif)*0.8

        # créer les répertoires test et train
        create_rep_split()

        # remplir les rep test et train
        for (nombre,phrase) in enumerate(corpus_negatif, start=1):
            fichier = f"{nombre}.txt"
            # mettre 80% de negatif dans train
            if nombre < train_neg_taille:
                with open(f"{path}/train/negatif/{fichier}",'w',encoding='utf8') as out:
                    out.write(phrase)
            # les fichiers restants pour le test (qui prend environ 20% des données)
            else:
                with open(f"{path}/test/negatif/{fichier}",'w',encoding='utf8') as out:
                    out.write(phrase)
                
        for (nombre,phrase) in enumerate(corpus_positif, start=1):
            fichier = f"{nombre}.txt"
            if nombre < train_pos_taille:
                with open(f"{path}/train/positif/{fichier}",'w',encoding='utf8') as out:
                    out.write(phrase)
            else:
                with open(f"{path}/test/positif/{fichier}",'w',encoding='utf8') as out:
                    out.write(phrase)


def load_datasets():
    """
        à partir du répertoire test et train, transformer les données à la forme de datasets demandée par sklearn
    """
    chemin = f"{path}/../ressources"
    X_data = {'train':[], 'test':[]}
    y = {'train':[], 'test':[]}

    for nom_type in ['train', 'test']:
        rep_corpus = os.path.join(chemin, nom_type)

        for label in ['positif','negatif']:
            rep_label = os.path.join(rep_corpus, label)
            liste_fichiers = os.listdir(rep_label)
            print("**{}**: label: {}, nombre: {}".format(nom_type, label, len(liste_fichiers)))

            for fname in liste_fichiers:
                path1 = os.path.join(rep_label, fname)
                with open(path1, 'r', encoding='utf8') as entree:
                    contenu = entree.read()
                X_data[nom_type].append(contenu)
                y[nom_type].append(label)

        print(" Le nombre de [{}] au total: {}\n".format(nom_type, len(X_data[nom_type])))
    return X_data['train'], y['train'], X_data['test'], y['test']


def importer_data_csv(fic_csv, test_size, lowercase, stopword):
    """
        une autre manière pour importer le data directement par fichier csv
    """
    path = sys.path[0]
    csvfile = f"{path}/../ressources/{fic_csv}"
    df = pd.read_csv(csvfile)

    stopwords = nor.importer_stopwords("stopwords_fr.txt")
    df['normalise'] = df["text"].apply(lambda x:nor.nettoyage(x, stopwords, lower=lowercase, stopword=stopword))
    X_train, X_test, y_train, y_test = train_test_split(df.normalise.values, df.label.values, test_size=test_size, stratify=df.label.values)
    # X_train, y_train, X_test, y_test = cross_validation.train_test_split(df.normalise.values, df.label.values, test_size=0.4, stratify=df.label.values)
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="créer des répertoires de data bien formés et normaliser le texte")
    parser.add_argument("taille", type=int, help="taille de l'échantillon")
    parser.add_argument("--Stop", type=bool, help="booléen, choisir si on traite les stopwords ou pas")
    parser.add_argument("--ficCsv", help="nom du fichier corpus csv")
    parser.add_argument("--lowercase", type=bool, help="donner le True si vous voulez mettre tous les mots en minuscules")
    
    args = parser.parse_args()
    
    size = args.taille
    stopword = args.Stop
    lowercase = args.lowercase
    fic_csv = "french_tweets.csv"
    if args.ficCsv:
        fic_csv = args.ficCsv
        

    
    nom_csv_sample = echantillonner(size, ficName=fic_csv)
    # exemple
    # nom_csv_sample = echantillonner(20, ficName="echantillon_200.csv")
    
    corpus_separation(nom_csv_sample, minuscule=lowercase, stopword=stopword)
    # d'autre choix possible
    # corpus_separation(nom_csv_sample, minuscule=True)
    # corpus_separation(nom_csv_sample, minuscule=True, stopword=True)   

if __name__ == "__main__":
    main()

