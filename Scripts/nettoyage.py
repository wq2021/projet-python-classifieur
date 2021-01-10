import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re 

def nettoyage(ligne):
    ligne = str(ligne)
    
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
    
    return ligne
nettoyage("http:www.bing.com oooups..., :des rires: quel :* red :o jour &amp; le travail http://www.bai.com !")