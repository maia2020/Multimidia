# Partie 1: Extraction de descripteurs
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# Calcul d'enveloppe d'énergie:
def enveloppe_energie(amplitude, taille):
    # On utilise comme arguments un vecteur contenant les amplitudes
    # et le nombre d'echantillons de la fenetre temporelle (taille)
    x_rms = []  # On initialize le vecteur des valeurs rms
    ite = 0  # Correspond à la position du premier élement de la fenêtre de calcul
    pas = taille // 2
    while taille + ite < len(amplitude):
        somme = (
            amplitude[ite : ite + taille]
        ) ** 2  # ite+taille = position du dernier élément de la fenêtre
        x_rms.append(np.sqrt((somme.sum()) / taille))
        ite = ite + pas  # On actualise le valeur de depart avec 50% de recouvrement
    return np.array(x_rms)


# Calcul du centroide temporel
# On utilisant l'enveloppe d'énergie on peut calculer des descripteurs comme le centroide temporel


def centroide_temporel(e):
    # On utilise comme argument l'enveloppe
    t = np.arange(1, e.shape[0] + 1)  # vecteur temporel
    tc = (e * t).sum() / e.sum()
    return tc


# Calcul de la durée effective
def duree_effective(e):
    # On utilise comme argument l'enveloppe
    # On suppose qu’un signal pouvant atteindre 20 % de l’intensité maximale est toujours important.
    seuil = np.max(e) * 0.2
    duree = len(np.where(e > seuil)[0])
    return duree


# Calcul de l'énergie globale
# Tout simplement la moyenne de l'enveloppe
def energie_globale(e):
    eglobale = e.mean()
    return eglobale


# Calcul du taux de passage par zéro
def zcr(s):
    sn = np.sign(s[0:-1])
    sn1 = np.sign(s[1:])
    taux = np.sum(np.abs(sn - sn1)) / 2
    return taux


# Partie 2: Indexation Sonore


# On commence pour normaliser les donnes
def normalisation(matrice):
    std = matrice.std(axis=0)
    moy = matrice.mean(axis=0)
    norm = np.linalg.norm(matrice)
    normalized_matrix = matrice / norm
    return normalized_matrix, moy, std


# On commence pour trouver le nombre d'écantillons de reference

donnees_app = np.loadtxt("donnees_app.txt")

# Initialisation
# Nombre d'échantillons de référence
nombrelignes = donnees_app.shape[0]
desc_app = np.zeros((nombrelignes, 4))

# Taille des fenêtres utilisées pour le calcul de l'enveloppe d'énergie
T = 300


# Calcul des descripteurs des échantillons de référence

for i in range(nombrelignes):
    s = donnees_app[i, :]  # Échantillon de référence
    N = len(s)
    e = enveloppe_energie(s, T)
    desc_app[i, 0] = centroide_temporel(e)
    desc_app[i, 1] = duree_effective(s)
    desc_app[i, 2] = energie_globale(e)
    desc_app[i, 3] = zcr(s)

# Après on normalize la matrice

desc_app_norm, desc_moy, desc_std = normalisation(desc_app)


# Alors pour l'indexation
donnees_test = np.loadtxt("donnees_test.txt")

# Initialisation
# Nombre d'échantillons de référence
nombrelignes = donnees_test.shape[0]
desc_test = np.zeros((nombrelignes, 4))

# Taille des fenêtres utilisées pour le calcul de l'enveloppe d'énergie
T = 300


# Calcul des descripteurs des échantillons de référence

for i in range(nombrelignes):
    s = donnees_test[i, :]  # Échantillon de référence
    N = len(s)
    e = enveloppe_energie(s, T)
    desc_test[i, 0] = centroide_temporel(e)
    desc_test[i, 1] = duree_effective(s)
    desc_test[i, 2] = energie_globale(e)
    desc_test[i, 3] = zcr(s)

# Après on normalize la matrice

desc_test_norm, desc_test_moy, desc_test_std = normalisation(desc_test)


# On calcule maintenet la distance entre les descripteurs d'apprentisage et ces du test

ligne_test = 1  # On choisit une ligne de la matrice de test


def calcul_distance(a, b, ligne_test):
    """On va calculer la distance entre la ligne_test de la matrice de test et toutes les lignes de la matrice d'apprentissage
    on va donc trouver les 5 distances les plus petites et on va les trier par ordre croissant en gardant les indices de la matrice d'apprentissage
    """
    distance = np.zeros((a.shape[0], 2))
    for i in range(a.shape[0]):
        distance[i, 0] = np.linalg.norm(a[i, :] - b[ligne_test, :])
        distance[i, 1] = i
    distance = distance[distance[:, 0].argsort()]
    return distance


# On calcule la distance entre les descripteurs d'apprentissage et ceux de test
distance = calcul_distance(desc_app_norm, desc_test_norm, ligne_test)

# On va maintenant trouver les 5 plus petites distances


def plus_petites_distances(distance, k):
    """On va trouver les k plus petites distances et on va les trier par ordre croissant en gardant les indices de la matrice d'apprentissage"""
    distance = distance[0:k, :]
    distance = distance[distance[:, 0].argsort()]
    return distance


# On calcule les 5 plus petites distances
k = 5
distance = plus_petites_distances(distance, k)

# On va alors recuperer les indices de la matrice d'apprentissage
indices = distance[:, 1]

# On va alors trouver dans les fichiers de noms les noms correspondants aux indices
noms_app = np.loadtxt("noms_app.txt", dtype=str)
noms_test = np.loadtxt("noms_test.txt", dtype=str)

# On va alors afficher les noms des 5 plus proches voisins
print("Les 5 plus proches voisins de", noms_test[ligne_test], "sont:")
for i in range(k):
    print(noms_app[int(indices[i])])

# Alors on va verifier les genres des 5 plus proches voisins dans le fichier "classes_app.txt"
classes_app = np.loadtxt("classes_app.txt", dtype=str)
classes_test = np.loadtxt("classes_test.txt", dtype=str)

# 1 : Classique
# 2 : Disco
# 3 : Jazz
# 4 : Rock

# On va alors afficher les genres des 5 plus proches voisins, mais avant on va les convertir en strings pour pouvoir les comparer
# On n'a pas vraiment besoin de cette prochaine partie, mais on l'a fait pour mieux comprendre le resultat
print("Les genres des 5 plus proches voisins de", noms_test[ligne_test], "sont:")
for i in range(k):
    if classes_app[int(indices[i])] == "1":
        print("Classique")
    elif classes_app[int(indices[i])] == "2":
        print("Disco")
    elif classes_app[int(indices[i])] == "3":
        print("Jazz")
    elif classes_app[int(indices[i])] == "4":
        print("Rock")

# On va alors choisir le genre le plus frequent et ça doit etre le genre de la musique de test
# On va utiliser la fonction Counter
genres = []
for i in range(k):
    if classes_app[int(indices[i])] == "1":
        genres.append("Classique")
    elif classes_app[int(indices[i])] == "2":
        genres.append("Disco")
    elif classes_app[int(indices[i])] == "3":
        genres.append("Jazz")
    elif classes_app[int(indices[i])] == "4":
        genres.append("Rock")

# On va alors dire que le genre de la musique de test est le genre le plus frequent et on va le comparer avec le genre de la musique de test
genre_test = Counter(genres).most_common(1)[0][0]
print("Le genre de la musique de test selon notre algo est:", genre_test)

# On va alors comparer avec le genre de la musique de test
# Mais avant on va convertir le genre de la musique de test en string
if classes_test[ligne_test] == "1":
    genre_test_vrai = "Classique"
elif classes_test[ligne_test] == "2":
    genre_test_vrai = "Disco"
elif classes_test[ligne_test] == "3":
    genre_test_vrai = "Jazz"
elif classes_test[ligne_test] == "4":
    genre_test_vrai = "Rock"

# On va alors comparer les deux genres
if genre_test == genre_test_vrai:
    print("Notre algorithme a bien predit le genre de la musique de test")
else:
    print("Notre algorithme n'a pas bien predit le genre de la musique de test")

print("-------------------------------------------------------------------")

# On va maintenant faire la meme chose pour toutes les musiques de test
# On va voir si l'algorithme a bien predit le genre de toutes les musiques de test et on va calculer le taux de succes
bien_predit = 0
for i in range(0, 24):
    distance = calcul_distance(desc_app_norm, desc_test_norm, i)
    distance = plus_petites_distances(distance, k)
    indices = distance[:, 1]
    genres = []
    for j in range(k):
        if classes_app[int(indices[j])] == "1":
            genres.append("Classique")
        elif classes_app[int(indices[j])] == "2":
            genres.append("Disco")
        elif classes_app[int(indices[j])] == "3":
            genres.append("Jazz")
        elif classes_app[int(indices[j])] == "4":
            genres.append("Rock")
    genre_test = Counter(genres).most_common(1)[0][0]
    if classes_test[i] == "1":
        genre_test_vrai = "Classique"
    elif classes_test[i] == "2":
        genre_test_vrai = "Disco"
    elif classes_test[i] == "3":
        genre_test_vrai = "Jazz"
    elif classes_test[i] == "4":
        genre_test_vrai = "Rock"
    if genre_test == genre_test_vrai:
        bien_predit = bien_predit + 1

print("Le taux de succes de notre algorithme est:", bien_predit / 24)


print("-------------------------------------------------------------------")

# On va maintenant voir l'évolution du taux de succes en fonction de k
# On va faire k varier de 1 à 10
# On va faire la meme chose pour toutes les musiques de test
# On va voir si l'algorithme a bien predit le genre de toutes les musiques de test et on va calculer le taux de succes
taxas_de_sucesso = []

# Loop para diferentes valores de k de 1 a 11
for k in range(1, 12):
    bien_predit = 0

    for i in range(0, 24):
        distance = calcul_distance(desc_app_norm, desc_test_norm, i)
        distance = plus_petites_distances(distance, k)
        indices = distance[:, 1]
        genres = []
        for j in range(k):
            if classes_app[int(indices[j])] == "1":
                genres.append("Classique")
            elif classes_app[int(indices[j])] == "2":
                genres.append("Disco")
            elif classes_app[int(indices[j])] == "3":
                genres.append("Jazz")
            elif classes_app[int(indices[j])] == "4":
                genres.append("Rock")
        genre_test = Counter(genres).most_common(1)[0][0]
        if classes_test[i] == "1":
            genre_test_vrai = "Classique"
        elif classes_test[i] == "2":
            genre_test_vrai = "Disco"
        elif classes_test[i] == "3":
            genre_test_vrai = "Jazz"
        elif classes_test[i] == "4":
            genre_test_vrai = "Rock"
        if genre_test == genre_test_vrai:
            bien_predit = bien_predit + 1

    taux_de_succes = bien_predit / 24
    taxas_de_sucesso.append(taux_de_succes)

# On va alors afficher le graphique
plt.plot(np.arange(1, 12), taxas_de_sucesso)
plt.xlabel("Valeur de k")
plt.ylabel("Taux de succes")
plt.title("Taux de succes en fonction de k")
plt.show()
