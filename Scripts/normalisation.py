# -*- coding: utf-8 -*-
# pour exécuter le script (voir un exemple de normalisation):
# python3 normalisation.py

import re 
import sys

# localiser le corpus, echantillon, repertoire créé
path = f"{sys.path[0]}/../ressources"               ##########################################


def importer_stopwords(nom_fic):
    """
        importer le fichier stopwords du répertoire ressources
        renvoyer un ensemble de stopwords
    """
    # assurer que le chemin relatif est par rapport au nettoyage.py donc marche n'importe où on lance le programme
    # chemin = sys.path[0]
    fic = f"{path}/{nom_fic}"
    # print(fic)

    stopwords = set()
    with open(fic, encoding="utf8") as f:
        for ligne in f:
            stopwords.add(ligne[:-1])
    return stopwords

def nettoyage(ligne, lower=False, stopword=None):
    """
        normalisation du texte tweet, renvoie la phrase après traitement; 
        cette fonction va supprimer les ponctuations, les symboles spéciaux et les urls, 
        remplacer les emoticons par des mots, et essayer de traiter un peu l'orthographe,
        mais le traitement de casse et suppression de stopwords est facultatif, pour le faire, utiliser comme : 
            nettoyage(phrase, lower=True, stopword = nom_du_fichier_stopwords)
            sinon:
            nettoyage(phrase)
    """
    ligne = str(ligne)
    
    # 1. remplacer plus de deux points par un point de suspension
    ligne = re.sub(r"\.{2,}","…",ligne)
    
    # 2. remplacer les symboles d'HTML par leurs symboles généraux
    ligne = re.sub(r"&\s?quot\s?;","\"",ligne)
    ligne = re.sub(r"&\s?amp\s?;","&",ligne)
    ligne = re.sub(r"&\s?lt\s?;","<",ligne)
    ligne = re.sub(r"&\s?gt\s?;",">",ligne)
    
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

    # 6. supprimer des ponctuations et des chiffres
    ligne = re.sub(r"[+@#&%!?\|\"{\(\[|_\)\]},\.;/:§”“‘~`\*]", "", ligne)
    ligne = re.sub(r"[0-9]+", "", ligne)
    
    # 7. supprimer des symboles spéciaux
    ligne = re.sub(r"♬|♪|♩|♫","",ligne)

    # 8. la répétition d'une lettre ou des lettres
    # Exemple d'un tweet: Ça va être un loooooooooooooooonnnnnnngggggggg ==> Ça va être un loonngg
    ligne = re.sub(r"((\w)\2{2,})",r"\2\2",ligne)

    # 9. ajouter une espace après l'apostrophe, pour la tokenization suivante
    ligne = re.sub(r"'",r"' ",ligne)
    ligne = re.sub(r"aujourd' hui",r"aujourd'hui",ligne)
    
    # 10. Traitement la casse et les stopwords selon parametres
    if lower:
        ligne = ligne.lower()
    if stopword:
        stopwords = importer_stopwords(stopword)
        lst_tokens = ligne.split()
        for token in lst_tokens:
            if token in stopwords:
                lst_tokens.remove(token)
        ligne = " ".join(lst_tokens).strip()

    return ligne


def main():
    #exemple
    # path_de_fic_stopwords = "projet-python-classifieur/ressources/stopwords_fr.txt"
    expre= "ne frequante pas http:www.bing.com ooooooups..., :des RIRES: quel :* red :o jour &amp; le travail http://www.bai.com !"
    expre="presque terminé vêtements beaucoup tasses décorées pas encore finies toujours mon enveloppe serviettes toilette faisant tâches … yay la maison seule"
    print(f"Exemple avant la normalisation: \n {expre}\nAprès la normalisation:")  
    print(nettoyage(expre, lower=True, stopword="stopwords_fr.txt"))

if __name__ == "__main__":
    main()
