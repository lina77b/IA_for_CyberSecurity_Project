# TP1 : Intelligence artificielle pour la cybersécurité

## Partie 1 : Découverte et préparation des données

---

## Travail demandé 1

1. **Jeu d'apprentissage** : 125 973 observations.
2. **Jeu de test** : 22 544 observations.
3. **Variables** : 43 colonnes au total, soit 41 caractéristiques de connexion (durée, protocole, service, nombre d'octets, taux d'erreurs…), plus la colonne `id` et la colonne cible `class`.
4. **Variable à prédire** : `class` (normal ou anomalie). Après regroupement des attaques, c'est `attack_category`, avec 5 classes.
5. **Rôle de `difficulty`** : elle indique à quel point une connexion est difficile à classer correctement. C'est une information sur l'observation, pas sur le trafic lui-même, donc on ne l'utilise pas comme variable explicative.

---

## Travail demandé 2

1. **Pourquoi les catégories ne passent pas directement** : KNN, SVM, la régression logistique et les réseaux de neurones font des calculs (distances, produits scalaires, gradients). Ils ne savent pas calculer avec du texte comme « tcp » ou « http ».
2. **Pourquoi TCP→0, UDP→1, ICMP→2 est un problème** : on crée un ordre et des distances qui n'existent pas. Le modèle croirait qu'ICMP est « plus grand » que TCP, ou qu'UDP est « entre » les deux, ce qui n'a aucun sens pour des protocoles.
3. **Encodage plus adapté** : le One-Hot Encoding.

---

## Travail demandé 3

1. **Classes équilibrées ?** Non. Normal et DoS dominent largement, tandis que R2L et surtout U2R sont très rares.
2. **Classe majoritaire** : Normal (suivie de DoS).
3. **Pourquoi le déséquilibre pose problème** : le modèle voit très peu d'exemples des classes rares, donc il les apprend mal. Il peut les ignorer sans que cela se voie dans les scores globaux.
4. **Pourquoi l'accuracy seule ne suffit pas** : elle compte juste la proportion de bonnes réponses, donc les classes majoritaires la dominent. Un IDS peut afficher 95 % d'accuracy tout en ratant presque toutes les attaques rares. Il faut regarder le recall, la precision et le F1 par classe.

**Question de réflexion** : oui, un modèle qui répond toujours « Normal » aurait environ 95 % d'accuracy. Mais il ne détecterait aucune attaque (recall de 0 % sur les attaques). Il paraît excellent sur le papier et est totalement inutile en pratique.

---

## Travail demandé 4

On code Normal=0, DoS=1, Probe=2, R2L=3, U2R=4, et on vérifie avec `y.unique()` que l'on obtient bien uniquement {0, 1, 2, 3, 4}.

| Classe | Code |
|--------|------|
| Normal | 0 |
| DoS    | 1 |
| Probe  | 2 |
| R2L    | 3 |
| U2R    | 4 |

---

## Travail demandé 5

1. **Principe du One-Hot** : chaque catégorie devient une colonne binaire. Si une connexion est en TCP, la colonne `protocol_type_tcp` vaut 1 et `udp` et `icmp` valent 0.
2. **Pourquoi c'est mieux qu'une numérotation** : toutes les catégories sont traitées de façon équivalente, sans ordre ni distance artificiels.
3. **Pourquoi les colonnes doivent être identiques entre train et test** : le modèle attend exactement les mêmes colonnes dans le même ordre. S'il en manque une (par exemple un service absent du test) ou si l'ordre change, les valeurs ne correspondent plus aux bons poids et les prédictions deviennent fausses, ou le modèle plante.

---

## Travail demandé 6

1. **Pourquoi un ensemble de validation** : pour comparer des modèles et régler les hyperparamètres (le k de KNN, le C du SVM…), et repérer le surapprentissage, sans toucher au test.
2. **Pourquoi pas le test** : si on choisit les réglages en regardant le test, on les adapte à ce jeu précis. Le score final devient trop optimiste et ne reflète plus la performance sur des données vraiment nouvelles.
3. **Validation vs test** : la validation sert à choisir et ajuster pendant le développement, on peut s'en servir plusieurs fois. Le test sert à juger une seule fois le modèle final.

---

## Travail demandé 7

**Pourquoi pas `fit_transform` sur le train puis sur le test** : le scaler doit calculer la moyenne et l'écart-type uniquement sur l'apprentissage, puis appliquer ces mêmes valeurs au test avec `transform`. Si on refait un `fit` sur le test :

- les informations du test influencent le prétraitement : c'est du **data leakage** ;
- train et test ne sont plus à la même échelle (μ et σ différents), donc le modèle reçoit des données incohérentes avec ce qu'il a appris ;
- en situation réelle, on ne connaît pas à l'avance les statistiques des futures connexions, donc l'évaluation serait irréaliste.

---

## Vérification finale

Avant d'entraîner, on contrôle :

- les dimensions de `X_train`, `X_val`, `X_test` et de `y_train`, `y_val`, `y_test` ;
- l'absence de valeurs non numériques ou manquantes ;
- la présence des 5 classes ;
- une moyenne proche de 0 et un écart-type proche de 1 sur les données d'apprentissage.

---

## À retenir

1. **Mes données** : des connexions réseau NSL-KDD (125 973 en train, 22 544 en test), décrites par 41 caractéristiques, dont 3 catégorielles (`protocol_type`, `service`, `flag`).
2. **Ma cible** : le type de trafic en 5 classes : Normal=0, DoS=1, Probe=2, R2L=3, U2R=4. `id` et `difficulty` ne sont pas des variables explicatives.
3. **Représentation** : One-Hot Encoding des catégories, mêmes colonnes dans train, validation et test, puis standardisation.
4. **Équilibre des classes** : non (Normal et DoS dominent, R2L et U2R sont rares). On regarde donc precision, recall et F1 macro, pas seulement l'accuracy.
5. **Séparation** : test fourni conservé tel quel, train divisé en 80 % / 20 % (stratifié) pour la validation. La validation sert à choisir les réglages, le test à l'évaluation finale.
6. **Data leakage** : `fit` du scaler uniquement sur `X_train`, puis `transform` sur validation et test. Jamais de réglage basé sur le test.

**En bref** : de bons résultats n'ont de sens que si les données sont bien comprises, bien encodées, bien séparées et sans fuite d'information.