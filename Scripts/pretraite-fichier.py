import sys
import numpy as np
# import csv
import pandas as pd
import normalisation as nor
from sklearn.model_selection import train_test_split

# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer

def importer_data_csv(fic_csv, test_size):
    path = sys.path[0]
    csvfile = f"{path}/../ressources/{fic_csv}"
    df = pd.read_csv(csvfile)
    # print(df.head())
    # print(df.label.unique())      #obtenir les cat

    stopwords = nor.importer_stopwords("stopwords_fr.txt")
    df['normalise'] = df["text"].apply(lambda x:nor.nettoyage(x, stopwords, lower=True, stopword=True))
    # print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df.normalise.values, df.label.values, test_size=test_size, stratify=df.label.values)
    # X_train, y_train, X_test, y_test = cross_validation.train_test_split(df.normalise.values, df.label.values, test_size=0.4, stratify=df.label.values)
    return X_train, y_train, X_test, y_test



def main():
    a,b,c,d = importer_data_csv("echantillon_20.csv", 0.3)
    print("train-set:")
    print(d)
    


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
