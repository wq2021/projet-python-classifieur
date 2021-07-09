# M2 TAL : Projet Classifieur de tweets français (Python)

Auteurs : Jianying Liu, Qi Wang.

**Cette documentation présente le projet dans le cadre du cours « Langage de script » en M2 TAL parcours Ingénierie Multilingue à l’Inalco.**

**La classification de textes est une des tâches fondamentales dans le traitement automatique des langues. Elle est basée sur les documents proposés en tant que l'entraînement, dans le but de pouvoir classifier de nouveaux documents avec des étiquettes(labels) prédéfinis. La classification de textes peut s'appliquer aux différentes tâches en pratique, tels que la détection de spams, l'analyse de sentiments, la classification automatique d'actualités, etc.**

**Dans ce projet, nous nous penchons sur la classification de tweets français au niveau du sentiment.**

## Données

Les données ont été choisies sur [Kaggle](https://www.kaggle.com/), où les jeux de données(datasets) sont en open source. Ils présentent 1,5 million de tweets en français et leur sentiment(étiquette) en binaire (0 pour négatif, 1 pour positif) sous format csv. Voici le lien pour y accéder et télécharger: [french-twitter-sentiment-analysis](https://www.kaggle.com/hbaflast/french-twitter-sentiment-analysis).

## Objectifs

L'objectif de notre projet consiste à réaliser une chaîne de traitement de classification de textes à l'aide de [scikit-learn](https://scikit-learn.org/stable/index.html). Afin de créer un classifieur de documents, nous allons implémenter plusieurs méthodes pour l'extraction des features de données textuelles et plusieurs algorithmes pour la classification.

## Méthodologies

Nous avons travaillé ensemble pour la rédaction de la documentation, aussi pour la création et le test du code.

La classification de textes basant sur Scikit-learn peut être divisée par les étapes suivantes :

- [1. Prétraitement des données textuelles](#1-prétraitement-des-données-textuelles)
- [2. Génération des données d'entraînement et de test](#2-génération-des-données-dentraînement-et-de-test)
- [3. Extraction des features des textes](#3-Extraction-des-features-des-textes)
- [4. Construction et évaluation des classifieurs](#4-Construction-et-évaluation-des-classifieurs)

### 1. Prétraitement des données textuelles

Les données sont stockées dans un fichier csv, chaque ligne commence par la polarité de sentiment (0 pour le sentiment négatif, 1 pour le sentiment positif), se suit par le contenu des tweets.

Voici quelques exemples pour une visualisation de la structure:

```
0,"Noooooooooooooooooooooooooooooooooooooo! Rafa est hors de wimbledon, je suis tellement éviscéré."
0,Les chatons vont bientôt. Des moments tristes. Je les aime trop
0,Mais je ne peux pas regarder le clip vidéo.
1,Shopping demain oxo
1,"Je ne sais pas si vous le savez, mais ... j'aime lire étrange, hein?"
1,"Aller à toi aujourd'hui si mes parents vont partir. D'abord, ils disent en partant à 11, puis 12 ... bien maintenant, il est 12 ans, et nous sommes encore là!"
```

Ensuite, nous avons effectué les tâches suivantes après l'observation de données:

- Suppression de lignes doublons avec la fonction `drop_duplicate()` dans `pandas`.
- Nettoyage des données :

  a. supprimer des urls et des symboles spéciaux (♬,♪,♩,♫, etc.)

  b. remplacer des symboles d'HTML par leurs symboles généraux (`&amp;` => `&`)

  c. remplacer des émoticônes par le mot correspondant (`;)` => `smile`, `:o` => `surprise`, etc.)

  d. supprimer des ponctuations dans les émoticônes semi-textuels (`:des rires:` => des rires, `::soupir::` => soupir, etc.)

  e. faire la tokenisation

  f. mettre tous les mots en minuscule (optionnel)

  g. filtrer des stopwords (optionnel)

### 2. Génération des données d'entraînement et de test

Après le nettoyage de données, nous avons séparé et exporté des données en mettant 80% pour le train, 20% pour le test.
La structure des répertoires se trouve ci-dessous :

```
resources
├─train
│  ├─negatif
│  │     ├─ 1.txt
│  │     ├─ 2.txt
│  │     ├─ ...
│  │     └─ ...
│  ├─positif
│  │     ├─ 1.txt
│  │     ├─ 2.txt
│  │     ├─ ...
│  └─    └─ ...
└─test
    ├─negatif
    │     ├─ *.txt
    │     ├─ *.txt
    │     ├─ ...
    │     └─ ...
    ├─positif
    │     ├─ *.txt
    │     ├─ *.txt
    │     ├─ ...
    └─    └─ ...

Un échantillon de 1000 tweets générés sont également fournis dans le répertoire `output`.

```
À partir des répertoires de donnés bien séparés et établis, nous avons extrait le contenu et leur étiquette, en produisant l'output pouvant être reconnu par `scikit-learn` pour l'apprentissage: `X_train`(données de l'entraînement), `y_train`(étiquettes de l'entraînement), `X_test`(données du test), `y_test`(étiquettes du test).

```python
X_train, y_train, X_test, y_test = load_datasets()
```

### 3. Extraction des features des textes

Le module [sklearn.feature_extraction](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction) se sert à extraire des features d'images et de textes. Nous nous concentrons sur le sous-module `sklearn.feature_extraction.text`, qui permet d'établir des vecteurs de features à partir des documents textuels. Deux sous-modules ont été implémentés dans notre projet :

- `feature_extraction.text.CountVectorizer`: convertir une collection de documents textuels à une matrice de fréquence de tokens.

- `feature_extraction.text.TfidfVectorizer`: convertir une collection de documents à une matrice de feature TF-IDF.

Nous prenons l'extraction du feature TF-IDF à titre d'exemple :

```python
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
words = tfidf_vectorizer.get_feature_names()
```

### 4. Construction et évaluation des classifieurs

**Benchmark : classifieur de Naive Bayes**

Le classifieur Naive Bayes est un benchmark idéal parmi un grand nombre de classifieurs proposés par `scikit-learn`, dont [`MultinomialNB`](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) s'adapte à la classification de textes.

Il suffit d'appliquer le sous-module d'extraction de features de texte `tfidf_vectorizer` (ou `CountVectorizer`) avec le classifieur dans le pipeline. Ensuite, la fonction `fit()` permet d'entraîner les données d'entraînement en prenant le premier argument comme les données, et le deuxième argument comme les étiquettes des données(labels). Après, nous pouvons faire la prédiction sur les données de test avec la fonction `predict()`.

`sklearn.metrics.classification_report` permet d'afficher l'évaluation de la classification.

```python
text_clf = Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print("Naive Bayes : MultinomialNB()")
print(classification_report(predicted, y_test))
```

Au niveau des classifieurs, nous avons choisi :

- `MultinomialNB()`: Naive Bayes
- `LogisticRegression()` : Logistic Regression
- `SGDClassifier()`:SVM
- `DecisionTreeClassifier()`: Decision Tree

## Implémentation des modules

- ### librairies utilisés

**`os`, `shutil`, `pandas`, `re`, `numpy`, `sklearn`**

- ### modules créés

**`normalisation`, `pre_traitement`, `testerClassifieur`**

## Résultats et discussions

En considérant le volume important des données, nous avons décidé de sélectionner un échantillon de **200,000** tweets pour tester nos classifieurs.

En général, les classifieurs choisies ont eu un résultat assez bon, malgré le fait que les valeurs de F-mesure ne soient pas très satisfaisantes.
`TfidfVectorizer` est la meilleure en tant que la méthode d'extraction des features, par rapport à `CountVectorizer`, notamment en comparant le classifieur de régression logistique (0,76 vs 0,75) et de l'arbre de décision(0,68 vs 0.67), mais l'écart reste très petit.

En ce qui concerne le classifieur, `LogisticRegression()` a eu globalement un meilleur résultat, par contre, `DecisionTreeClassifier()` n'a toujours pas eu de bons résultats que d'autres classifieurs. Par conséquent, nous concluons que **l'algorithme `LogisticRegression()` avec le feature `TfidfVectorizer`** ont eu une meilleure performance pour nos données.

Voici les résultats en détails que nous avons obtenus :

### Méthode d'extraction des features de textes : `TfidfVectorizer`

```
Naive Bayes : MultinomialNB() tfidf
              precision    recall  f1-score   support

           0       0.77      0.73      0.75     32067
           1       0.71      0.75      0.73     27933

    accuracy                           0.74     60000
   macro avg       0.74      0.74      0.74     60000
weighted avg       0.74      0.74      0.74     60000
```

```
Logistic Regression : LogisticRegression() tfidf
              precision    recall  f1-score   support

           0       0.74      0.77      0.75     29274
           1       0.77      0.74      0.76     30726

    accuracy                           0.76     60000
   macro avg       0.76      0.76      0.76     60000
weighted avg       0.76      0.76      0.76     60000
```

```
SVM : SGDClassifier() tfidf
              precision    recall  f1-score   support

           0       0.70      0.78      0.74     27538
           1       0.79      0.72      0.76     32462

    accuracy                           0.75     60000
   macro avg       0.75      0.75      0.75     60000
weighted avg       0.75      0.75      0.75     60000
```

```
Decision Tree : DecisionTreeClassifier() tfidf
              precision    recall  f1-score   support

           0       0.68      0.68      0.68     30327
           1       0.67      0.67      0.67     29673

    accuracy                           0.68     60000
   macro avg       0.68      0.68      0.68     60000
weighted avg       0.68      0.68      0.68     60000
```

### Méthode d'extraction des features de textes : `CountVectorizer`

```
Naive Bayes : MultinomialNB() countVectorizer
              precision    recall  f1-score   support

           0       0.77      0.74      0.75     31491
           1       0.72      0.75      0.74     28509

    accuracy                           0.74     60000
   macro avg       0.74      0.74      0.74     60000
weighted avg       0.74      0.74      0.74     60000
```

```
Logistic Regression : LogisticRegression() countVectorizer
              precision    recall  f1-score   support

           0       0.73      0.76      0.75     29109
           1       0.77      0.74      0.75     30891

    accuracy                           0.75     60000
   macro avg       0.75      0.75      0.75     60000
weighted avg       0.75      0.75      0.75     60000
```

```
SVM : SGDClassifier() countVectorizer
              precision    recall  f1-score   support

           0       0.71      0.78      0.75     27704
           1       0.80      0.73      0.76     32296

    accuracy                           0.75     60000
   macro avg       0.75      0.76      0.75     60000
weighted avg       0.76      0.75      0.75     60000
```

```
Decision Tree : DecisionTreeClassifier() countVectorizer
              precision    recall  f1-score   support

           0       0.69      0.67      0.68     31298
           1       0.65      0.67      0.66     28702

    accuracy                           0.67     60000
   macro avg       0.67      0.67      0.67     60000
weighted avg       0.67      0.67      0.67     60000
```

Après une observation des résultats obtenus, nous soutaitons de proposer quelques pistes d'amélioration. En premier lieu, nous espérons optimiser les hyperparamètres des classifieurs. Nous avons essayé de voir les hyperparamètres des classifieurs testés sur `Scikit-learn`, mais nous avons eu mal à comprendre et choisir un meilleur hyperparamètre. Grâce à un article concernant exactement la modification des hyperparamètres de classifieurs que nous avons choisis pour ce projet, nous avons pu tester et obtenir les performances pour quelques classifeurs ayant différents paramètres, mais cela n'est pas suffisant pour d'autres classifieurs.

D'ailleurs, il faudrait mieux créer un feature par nous-même, au lieu de choisir directement un feature proposé par `scikit-learn`. Par exemple, les traits linguistiques tels que la longueur moyenne des mots, le nombre des points de ponctuation peuvent être une bonne idée.

## Difficultés rencontrées

**1. Choix de la tokenisation pour le corpus français**

 Nous avons essayé d'utiliser deux librairies pour tokeniser les données français : [spaCy](https://spacy.io/) et [NLTK](https://www.nltk.org/).

 Néanmoins, NLTK ne peut pas proposer une tokenisation correcte, c'est-à-dire qu'il tokenise seulement en fonction des espaces. Par exemple, il ne tokenise pas le mot `c'est` en deux tokens `c'` et `est`. De plus, spaCy nous a pris trop de temps pour la tokenisation. 

 Par conséquent, nous proposons une méthode "stupide" pour effectuer la tokenisation avec de l'espace : nous ajoutons un espace une fois l'apostrophe `'` est apparu, sauf un cas particulier `aujourd'hui` en français.

 ```python
    ligne = re.sub(r"'",r"' ",ligne)
    ligne = re.sub(r"aujourd' hui",r"aujourd'hui",ligne)
 ```

**2. Utilisation du `Pandas`**

  Étant donnée que `sklearn` peut utiliser les données sous forme de `DataFrame` et notre corpus original est sous format `csv`, notre première idée est de structurer les données et d'utiliser seulement le document `csv` à l'aide de `pandas`. Mais ces deux librairies sont toutes nouvelles pour nous.
  
  Au début, nous n'avons pas trouvé un exemple dotant d'une structure de données similaire pour nous inspirer à lier ces deux formats. Donc nous nous sommes détournées vers la génération d'un répertoire de corpus. Après, un blog nous a montré un bon exemple, donc nous avons pu ajouter une fonction qui marchait plus efficacement, soit `importer_data_csv`.

**3. Filtrage des stopwords**

  Pendant l'élimination des mots grammaticaux, nous avons eu une observation: certains mots dans la liste `stopwords_fr.txt` ne sont pas supprimés après le traitement.

  Après beaucoup de temps bien circulé, nous avons finalement trouvé la raison : l'utilisation de `remove()`. En tant que solution, on a dû créer une nouvelle liste ou une chaîne de caractère vide pour rajouter et concaténer les mots filtrés.
