## Réponses aux questions de la Partie 2

Les chiffres viennent des données (jeu de test, KDDTest+). Les temps ont été mesurés sur une machine à 1 processeur et seront différents sur une autre machine, mais l'ordre entre modèles reste le même.

---

### KNN

**1. Pourquoi KNN est-il sensible à l'échelle des variables ?**

KNN classe selon la distance entre observations. Une variable à grande échelle (`src_bytes`, jusqu'à plusieurs millions) domine la distance et les variables à petite échelle (taux entre 0 et 1) sont ignorées. La standardisation met toutes les variables sur le même pied.

**2. Effet d'un k trop petit ?**

Le modèle est très sensible au bruit et aux points isolés : la frontière est irrégulière, avec un risque de surapprentissage (variance élevée).

**3. Effet d'un k trop grand ?**

Le vote est trop lissé : les classes minoritaires (R2L, U2R) sont « noyées » par les classes majoritaires, avec un risque de sous-apprentissage (biais élevé). Dans le notebook, les performances baissent bien quand k augmente (F1 sur validation : 0.9969 pour k = 3, 0.9921 pour k = 15).

**4. Quelle métrique pour choisir k, et pourquoi ?**

Le **F1 macro**, parce qu'il donne le même poids à chacune des 5 classes, y compris R2L et U2R qui sont rares. Le calcul avec le F1 macro sur la validation donne :

| k | 3 | 5 | 7 | 11 | 15 |
|---|---|---|---|---|---|
| F1 macro | **0.902** | 0.871 | 0.888 | 0.854 | 0.842 |

Le choix **k = 3** est confirmé.

---

### Naive Bayes

**1. Que signifie l'hypothèse « naïve » ?**

Le modèle suppose que, **sachant la classe**, les variables sont indépendantes les unes des autres. Cela permet de calculer P(X|C) comme un produit de probabilités simples : P(X|C) = ∏ P(xᵢ|C).

**2. Cette hypothèse est-elle réaliste pour des données réseau ?**

Non. Beaucoup de variables sont fortement corrélées : `serror_rate` et `srv_serror_rate`, `count` et `srv_count`, `dst_host_*`, etc. De plus, le One-Hot Encoding crée des colonnes binaires dépendantes entre elles (`service_http` exclut `service_ftp`). Enfin, `GaussianNB` suppose des variables de loi normale, ce qui est faux pour les colonnes binaires et les compteurs très asymétriques.

**3. Pourquoi le modèle peut-il malgré tout être efficace ?**

Pour classer, il suffit que la bonne classe obtienne la plus grande probabilité, même si les probabilités elles-mêmes sont mal estimées. Le modèle est aussi très rapide, simple, et il fonctionne avec peu de données. Sur ce jeu, il est **peu efficace** : accuracy 0.444 et F1 macro 0.313, avec par exemple seulement 10 % des Probe détectées. C'est cohérent avec l'hypothèse violée.

---

### Régression logistique

**1. Rôle du paramètre C ?**

C est l'inverse de la force de régularisation. Un **C petit** donne une régularisation forte (poids contraints, modèle simple, risque de sous-apprentissage). Un **C grand** donne une régularisation faible (le modèle colle aux données, risque de surapprentissage).

**2. Pourquoi la régularisation est-elle utile ?**

Elle pénalise les poids trop grands, ce qui limite le surapprentissage et stabilise le modèle quand il y a beaucoup de variables corrélées (123 colonnes ici après le One-Hot).

**3. Pourquoi `max_iter` peut-il être augmenté ?**

L'optimiseur (`lbfgs`) est itératif et s'arrête à 100 itérations par défaut. Sur 123 variables, il n'a pas toujours convergé (`ConvergenceWarning`). Augmenter `max_iter` (par exemple 1000) lui laisse le temps de converger vers la solution optimale.

---
### SVM (Support Vector Machine)

**1. Quel est le rôle de C ?**

C règle le compromis entre une **marge large** et le **nombre d'erreurs tolérées** sur les données d'apprentissage.

- **C petit** : forte régularisation, marge large, on tolère des erreurs. Le modèle est simple et risque le sous-apprentissage. C'est le cas pour C = 0.1 (F1 macro de seulement 0.70 sur la validation).
- **C grand** : faible régularisation, le modèle cherche à bien classer chaque point d'apprentissage. La frontière est plus complexe et il y a un risque de surapprentissage.

Résultats sur la validation (sous-échantillon de 20 000 observations) :

| C | Accuracy | Recall macro | F1 macro |
|---|---|---|---|
| 0.1 | 0.974 | 0.663 | 0.701 |
| 1 | 0.990 | 0.856 | 0.881 |
| 10 | 0.994 | 0.873 | 0.897 |

La configuration retenue est **C = 10** (meilleur F1 macro sur la validation).

**2. Que signifie l'utilisation d'un noyau RBF ?**

Le noyau RBF (Radial Basis Function) mesure la **similarité entre deux observations selon leur distance** :

K(x, x') = exp(−γ ‖x − x'‖²)

Il revient à projeter implicitement les données dans un espace de très grande dimension, où une séparation linéaire devient possible. Dans l'espace d'origine, cela donne une **frontière de décision non linéaire**, capable de s'adapter à des classes de forme complexe. Le paramètre γ règle la portée de chaque point : γ grand donne une frontière très locale (risque de surapprentissage), γ petit donne une frontière plus lisse. Ici `gamma="scale"` est utilisé.

**3. Pourquoi la standardisation est-elle particulièrement importante pour SVM ?**

Le noyau RBF repose sur des **distances**. Sans standardisation, une variable à grande échelle (comme `src_bytes`, jusqu'à plusieurs millions) domine le calcul de distance, et les variables à petite échelle (taux entre 0 et 1) sont quasiment ignorées. La standardisation met toutes les variables sur le même pied. Elle accélère aussi la convergence de l'optimisation.

**4. Quels problèmes pratiques peuvent apparaître lorsque le nombre d'observations est important ?**

- **Apprentissage lent** : la complexité est comprise entre O(n²) et O(n³). Avec ~100 000 observations, c'est la méthode la plus coûteuse à l'apprentissage (≈ 21 s sur le jeu complet, contre moins d'une seconde pour l'arbre de décision).
- **Mémoire** : le calcul du noyau entre paires d'observations devient très coûteux.
- **Prédiction lente** : elle dépend du nombre de vecteurs de support (≈ 5 s sur le test, contre 0.01 s pour l'arbre).
- **Réglage des hyperparamètres coûteux** : chaque essai (C, γ) demande un entraînement complet. C'est pourquoi C a été réglé sur un sous-échantillon de 20 000 observations.

**Résultats sur le jeu de test (C = 10)** : accuracy 0.782, precision macro 0.715, recall macro 0.610, F1 macro 0.603.

- Le F1 macro passe de 0.90 (validation) à 0.60 (test). La validation est tirée de KDDTrain+, donc très proche de l'apprentissage, alors que KDDTest+ contient des attaques absentes de l'apprentissage.
- **R2L** : seulement 500 connexions sur 2885 détectées (recall 0.17), et 2107 sont classées « Normal ».
- **U2R** : 25 sur 67 détectées (recall 0.37, F1 0.34).
- Normal, DoS et Probe sont bien mieux détectées.

---

### Arbre de décision

Comparaison selon la profondeur maximale (apprentissage vs validation) :

| max_depth | F1 macro apprentissage | F1 macro validation | Écart | Accuracy apprentissage | Accuracy validation |
|---|---|---|---|---|---|
| 5 | 0.721 | 0.722 | ≈ 0 | 0.984 | 0.982 |
| 10 | 0.937 | 0.903 | 0.034 | 0.998 | 0.996 |
| 15 | 0.982 | 0.921 | 0.061 | 0.999 | 0.998 |
| 20 | 0.9998 | 0.945 | 0.054 | 0.99995 | 0.998 |

La profondeur retenue est **max_depth = 20** (meilleur F1 macro sur la validation).

**1. Que se passe-t-il lorsque la profondeur augmente ?**

L'arbre pose plus de questions successives et découpe l'espace en régions de plus en plus petites : il devient **plus complexe**.

- Le score d'apprentissage augmente fortement (F1 macro de 0.72 à 0.9998).
- Le score de validation augmente aussi, mais moins vite (de 0.72 à 0.945).
- Le temps d'apprentissage et la taille du modèle augmentent.
- À max_depth = 5, l'arbre est trop simple (sous-apprentissage) : F1 macro de 0.72 à la fois sur l'apprentissage et sur la validation.

**2. Comment reconnaître un surapprentissage ?**

Par un **écart important et croissant entre le score d'apprentissage et le score de validation (ou de test)** : le modèle est presque parfait sur les données vues mais moins bon sur les données nouvelles.

Ici, l'écart de F1 macro est nul à max_depth = 5, puis passe à 0.034 (profondeur 10), 0.061 (profondeur 15) et 0.054 (profondeur 20). L'écart est encore plus net avec le jeu de test : F1 macro de 0.9998 sur l'apprentissage contre 0.645 sur le test. Le modèle généralise mal aux nouvelles attaques.

**3. Pourquoi la performance d'apprentissage peut-elle continuer à augmenter alors que celle de validation diminue ?**

Un arbre profond crée des **feuilles très spécifiques**, presque dédiées à chaque observation d'apprentissage, y compris le bruit et les cas particuliers. Ces règles améliorent le score sur les données déjà vues (l'arbre les mémorise) mais **ne se généralisent pas** à de nouvelles données, d'où la stagnation ou la baisse sur la validation.

Dans nos résultats, la validation n'a pas baissé : elle est tirée du même fichier KDDTrain+ que l'apprentissage et contient beaucoup de connexions quasi identiques, ce qui masque le surapprentissage. Il apparaît clairement sur le jeu de test.

**Résultats sur le jeu de test (max_depth = 20)** : accuracy 0.786, precision macro 0.740, recall macro 0.619, F1 macro 0.645. Temps d'apprentissage < 1 s et temps de prédiction ≈ 0.01 s : c'est le modèle le plus rapide et le meilleur en F1 macro.

- **R2L** : 976 détectées sur 2885 (recall 0.34), 1653 classées « Normal ».
- **U2R** : 23 sur 67 détectées (recall 0.34).
- **DoS** : 1606 connexions sur 7460 sont classées « Normal » (faux négatifs).

---

### Comparaison des 5 modèles (section 20)

Tableau de synthèse sur le test :

| Modèle | Accuracy | Precision macro | Recall macro | F1 macro | Temps apprentissage | Temps prédiction |
|---|---|---|---|---|---|---|
| KNN (k = 3) | 0.752 | 0.708 | 0.562 | 0.564 | 0.1 s | 16.3 s |
| Naive Bayes | 0.444 | 0.498 | 0.419 | 0.313 | 0.2 s | 0.04 s |
| Régression logistique (C = 10) | 0.778 | 0.639 | 0.654 | 0.592 | 5.1 s | 0.01 s |
| SVM (RBF, C = 10) | 0.782 | 0.715 | 0.610 | 0.603 | 20.8 s | 5.2 s |
| Arbre de décision (max_depth = 20) | 0.786 | 0.740 | 0.619 | 0.645 | 0.6 s | 0.01 s |

**1. Les modèles donnent-ils des performances comparables ?**

Quatre d'entre eux, oui : accuracy entre 0.75 et 0.79 et F1 macro entre 0.56 et 0.65. **Naive Bayes** est nettement en dessous (0.44). Le meilleur est l'arbre de décision (F1 macro 0.645), puis la SVM.

**2. Les performances sont-elles similaires pour les cinq classes ?**

Non. **Normal** (recall 0.93 à 0.97 sauf Naive Bayes) et **DoS** (F1 de 0.85 à 0.90) sont bien détectées, **Probe** moyennement (F1 de 0.65 à 0.79). **R2L** (F1 de 0.15 à 0.50) et **U2R** (F1 d'environ 0.34) sont très mal détectées.

**3. Quelle classe est la plus difficile à détecter ?**

**R2L** : son recall n'est que de 0.09 à 0.34 (KNN 0.10, régression logistique 0.09, SVM 0.17, arbre 0.34). **U2R** est aussi difficile (précision faible, 0.23 à 0.37 sauf Naive Bayes à 0.04), mais avec seulement 67 exemples dans le test. Ces deux attaques ressemblent à du trafic normal, sont rares dans l'apprentissage, et le test contient des variantes absentes du train.

**4. Quelle classe produit le plus de faux négatifs ?**

**R2L** pour le KNN (2588 connexions R2L non détectées sur 2885), la régression logistique (2635), la SVM (2385) et l'arbre (1909). Presque toutes sont classées « Normal ». Pour **Naive Bayes**, ce sont plutôt DoS (4290) et Normal (4092). En proportion, U2R est aussi très touchée (jusqu'à 69 % de non-détections pour la SVM et 66 % pour l'arbre).

**5. Quel modèle est le plus coûteux en temps de calcul ?**

- **À l'apprentissage** : la **SVM** (≈ 21 s), puis la régression logistique (≈ 5 s). KNN, Naive Bayes et l'arbre sont quasi instantanés.
- **À la prédiction** : le **KNN** (≈ 16 s), car il compare chaque observation de test à tout le jeu d'apprentissage. Il est suivi de la SVM (≈ 5 s). Les trois autres prédisent en moins d'une seconde.

**6. Pourquoi l'accuracy ne doit-elle pas être utilisée seule ?**

Les classes sont déséquilibrées (U2R : 67 exemples sur 22 544). Un modèle peut avoir une bonne accuracy en détectant bien Normal et DoS tout en ratant presque toutes les attaques rares. Exemple : la SVM a 78 % d'accuracy mais ne détecte que 17 % des R2L. Il faut regarder le recall et le F1 par classe, et les moyennes macro.

**Question de réflexion : conséquence d'un faux négatif dans un IDS ?**

Un faux négatif est une **attaque classée comme normale** : aucune alerte n'est levée, l'attaquant peut continuer. Pour R2L (accès distant non autorisé) et U2R (élévation de privilèges vers root), cela veut dire une machine compromise, des données volées, ou une porte dérobée installée dans le réseau IoT, sans que personne le sache. À l'inverse, un faux positif n'entraîne qu'une fausse alerte à vérifier, ce qui coûte du temps mais ne laisse pas d'intrusion. Dans un IDS, on cherche donc à maximiser le **recall des classes d'attaque**, même au prix de quelques fausses alertes en plus. Les résultats montrent que les modèles testés restent insuffisants sur ce point pour R2L et U2R.