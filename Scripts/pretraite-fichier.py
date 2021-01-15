import sys
import numpy as np
# import csv
import pandas as pd
import normalisation as nor

# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer



def echantillonner(taille, ficName="french_tweets.csv"):
    """
        échantillonnage aléatoire du corpus original, 
        créer un nouveau csv dans le répertoire ressources, imprimer dans le terminal l'info de l'échantillon

        nom du fichier sorti:  echantillon_taille.csv      ex. echantillon_100.csv
        taille du corpus original:
            0(negatif) 771604
            1(positif) 755120
    """
    # fixer le répertoire du corpus et l'adresse du fichier sortie
    path = sys.path[0]

    fichier = f"{path}/../ressources/{ficName}"
    df = pd.read_csv(fichier)

    # supprimer les doublons
    df2 = df.drop_duplicates()
    df2 = df2.sample(n=taille)

    print("################\nInfo sur les catégories:")
    print(df2["label"].value_counts())
    print("################")

    df2.to_csv(f"{path}/../ressources/echantillon_{taille}.csv", index=False)


def main():
    echantillonner(20)
    # path = sys.path[0]
    # csvfile = f"{path}/../ressources/echantillon_10.csv"
    # df = pd.read_csv(csvfile)
    # df2 = df["text"].apply(lambda x:nor.nettoyage(x, lower=True))
    # print(df["label"])
    # print(df2.values)
    # texte_globale = df2.values

    # colum_trans = ColumnTransformer(["category", ])



if __name__ == "__main__":
    main()






# print(df.head()["label"])
# print(df2["label"].value_counts())
# print(df2["text"].apply(lambda x:nor.nettoyage(x)))
# label_set = set()
# negatif = set()
# positif = set()

# with open(fichier, encoding="utf8") as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
#     for row in reader:
#         if row["label"] = "0":
#             negatif.add(row["text"])
#         elif row[]
