import re 
import os
from spacy.lang.fr import French

def tokenisation(contenu):
    chaine = ""
    nlp = French()
    doc = nlp(contenu)
    for token in doc:
        chaine += token.text
        chaine += ' '
    return chaine

def filtre_stopwords(phrase):
    liste_stopwords = []
    phrase_out = ""
    with open('stopwords_fr.txt','r',encoding='utf8') as entree:
        for ligne in entree:
            # afin d'éliminer "\n" à la fin de chaque stopword
            liste_stopwords.append(ligne[:-1])
    for mot in phrase.split(' '):
        if mot not in liste_stopwords:
            phrase_out += mot
            phrase_out += " "
    return phrase_out


def nettoyage(ligne, lower=False):
    
    # 1. remplacer plus de deux points par un point de suspension
    ligne = re.sub(r"\.{2,}","…",ligne)
    
    # 2. remplacer les symboles d'HTML par leurs symboles généraux
    ligne = re.sub(r"&quot;","\"",ligne)
    ligne = re.sub(r"&amp;","&",ligne)
    ligne = re.sub(r"&lt;","<",ligne)
    ligne = re.sub(r"&gt;",">",ligne)
    
    # 3 . supprimer des urls
    ligne = re.sub(r'http:/?/?.\S+',r'',ligne)

    # 4 . remplacer les emoticons
    ligne = re.sub(r':‑\)|;\)|;-\)|:\)|:\'\)|:]|;]|:-}|=]|:}|\(:',"smile",ligne)
    ligne = re.sub(r'XD|XD{2,}',"laugh",ligne)
    ligne = re.sub(r':\(|:\[|:\'-\(|:\'\(',"sad",ligne)
    ligne = re.sub(r':o',"surprise",ligne)
    ligne = re.sub(r':X|:-\*|:\*',"kiss",ligne)
    ligne = re.sub(r':\|',"indecision",ligne)
    ligne = re.sub(r':X|:-X',"togue-tied",ligne)
    ligne = re.sub(r':-/|:/|:\|:L/:S',"annoyed",ligne)
    
    # 5. remplacer des emotions semi-textuels, par exemples: :des rires:, ::soupir::, etc.
    ligne = re.sub(r'\:{1,}(\w+\s?\w+?)\:{1,}',r'\1',ligne)

    # 6. supprimer des ponctuations
    ligne = re.sub(r"[+@#&%!?\|\"{\(\[|_\)\]},\.;/:§”“‘~`\*]", "", ligne)
    
    # 7. supprimer des symboles spéciaux
    ligne = re.sub(r"♬|♪|♩|♫","",ligne)

    # 8. la répétition d'une lettre ou des lettres
    # Exemple d'un tweet: Ça va être un loooooooooooooooooooooooooooooooonnnnnnngggggggg
    ligne = re.sub(r"((\w)\2{2,})",r"\2\2",ligne)
    
    # 9. mettre les mots en minuscule 
    if lower:
        ligne = ligne.lower()
    return ligne
    

def corpus_separation():
    with open('french_tweets.csv','r',encoding="utf8") as entree:
        corpus_negatif = []
        corpus_positif = []
        for ligne in entree:
            if ligne.startswith('0'):
                phrase = ligne[2:]
                phrase = nettoyage(phrase)
                phrase = tokenisation(phrase)
                phrase = filtre_stopwords(phrase)
                phrase = phrase.strip()
                corpus_negatif.append(phrase)
                
            if ligne.startswith('1'):
                phrase = ligne[2:]
                phrase = nettoyage(phrase)  
                phrase = tokenisation(phrase)                 
                phrase = filtre_stopwords(phrase)
                phrase = phrase.strip()               
                corpus_positif.append(phrase)
        
        for (nombre,phrase) in enumerate(corpus_negatif, start=1):
            fichier = str(nombre) + str(".txt")
            if nombre < 600001:
                with open(os.path.join('/Users/wq/Desktop/TAL/M2/S1/Python/Projet_final/data/train/negatif',fichier),'w',encoding='utf8') as out:
                    out.write(phrase)
            else:
                with open(os.path.join('/Users/wq/Desktop/TAL/M2/S1/Python/Projet_final/data/test/negatif',fichier),'w',encoding='utf8') as out:
                    out.write(phrase)
                
        for (nombre,phrase) in enumerate(corpus_positif, start=1):
            fichier = str(nombre) + str(".txt")
            if nombre < 600001:
                with open(os.path.join('/Users/wq/Desktop/TAL/M2/S1/Python/Projet_final/data/train/positif',fichier),'w',encoding='utf8') as out:
                    out.write(phrase)
            else:
                with open(os.path.join('/Users/wq/Desktop/TAL/M2/S1/Python/Projet_final/data/test/positif',fichier),'w',encoding='utf8') as out:
                    out.write(phrase)
corpus_separation()

def load_datasets():

    chemin = '/Users/wq/Desktop/TAL/M2/S1/Python/Projet_final/data/'
    X_data = {'train':[], 'test':[]}
    y = {'train':[], 'test':[]}

    for nom_type in ['train', 'test']:
        rep_corpus = os.path.join(chemin, nom_type)

        for label in ['positif','negatif']:
            rep_label = os.path.join(rep_corpus, label)
            liste_fichiers = os.listdir(rep_label)
            print("**{}**: label: {}, nombre: {}".format(nom_type, label, len(liste_fichiers)))

            for fname in liste_fichiers:
                path = os.path.join(rep_label, fname)
                with open(path, 'r', encoding='utf8') as entree:
                    contenu = entree.read()
                X_data[nom_type].append(contenu)
                y[nom_type].append(label)

        print(" Le nombre de [{}] au total: {}\n".format(nom_type, len(X_data[nom_type])))
    
    return X_data['train'], y['train'], X_data['test'], y['test']
load_datasets()
