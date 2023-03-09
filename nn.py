### Réseau simple from scratch - Classification à 10 classes - Protech

### librairies
from math import *
from random import *
import matplotlib.pyplot as plt
from azureml.opendatasets import MNIST
import random
import datetime


### Fonctions annexes

def MSE(X_tot,Y_tot):
    somme = 0
    m = len(X_tot)
    for j in range(m):
        X = X_tot[j]
        Y = Y_tot[j]
        n = len(X)
        for i in range(n):
            somme = somme + (X[i] - Y[i])**2
    return(somme/(n*m))


def sortie_reseau(X):

    noeuds = []
    noeuds_in = []
    for k in range(neurones_cachees):
        noeuds_1_in = poids_couche_1[k][neurones_entrees]
        for j in range(neurones_entrees-1):
            noeuds_1_in = noeuds_1_in + X[j] * poids_couche_1[k][j]
        noeuds_1 = fonction_activation(noeuds_1_in)
        noeuds = noeuds + [noeuds_1]
        noeuds_in = noeuds_in + [noeuds_1_in]
    sortie_in = []
    sortie = []
    for k in range(n_out):
        noeuds_2_in = poids_couche_2[k][neurones_cachees]
        for j in range(neurones_cachees):
            noeuds_2_in = noeuds_2_in + noeuds[j] * poids_couche_2[k][j]
        noeuds_2 = fonction_activation(noeuds_2_in)
        sortie_in = sortie_in + [noeuds_2_in]
        sortie = sortie + [noeuds_2]

    return(sortie)

def moyenne(X):
    somme = 0
    for i in range(len(X)):
        somme = somme + X[i]
    return somme/len(X)

def nombre_element(L,a):
    n = len(L)
    compteur = 0
    for i in range(n):
        if L[i] == a:
            compteur = compteur + 1
    return(compteur)

def rang_valeur(X,a):
    i = -1
    for j in range(len(X)):
        if X[j] == a :
            i = j
    return(i)

def rang_max(X):
    return(rang_valeur(X,max(X)))

def creation_sortie(rang,n_out):
    l = []
    for i in range(n_out):
        if i == rang :
            l = l + [1]
        else :
            l = l + [0]
    return l

def affichage_matrice(X):
    n = len(X)
    for i in range(n):
        print(X[i])

### Importation MNIST
mnist = MNIST.get_tabular_dataset()
mnist_df = mnist.to_pandas_dataframe()
mnist_df.info()
data = mnist_df.drop("label", axis=1).astype(int).values/255.0
label = mnist_df.filter(items=["label"]).astype(int).values

### Récupération des images avec des 1 et des 0
nb_image_MNIST = len(label)
image = []
label_image = []
for i in range(nb_image_MNIST):
    if int(label[i]) == 0:
        image = image + [data[i]]
        label_image = label_image + [[1,0,0,0,0,0,0,0,0,0]]
    if int(label[i]) == 1:
        image = image + [data[i]]
        label_image = label_image + [[0,1,0,0,0,0,0,0,0,0]]
    if int(label[i]) == 2:
        image = image + [data[i]]
        label_image = label_image + [[0,0,1,0,0,0,0,0,0,0]]
    if int(label[i]) == 3:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,1,0,0,0,0,0,0]]
    if int(label[i]) == 4:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,1,0,0,0,0,0]]
    if int(label[i]) == 5:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,0,1,0,0,0,0]]
    if int(label[i]) == 6:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,0,0,1,0,0,0]]
    if int(label[i]) == 7:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,0,0,0,1,0,0]]
    if int(label[i]) == 8:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,0,0,0,0,1,0]]
    if int(label[i]) == 9:
        image = image + [data[i]]
        label_image = label_image + [[0,0,0,0,0,0,0,0,0,1]]



### On divise en données train et test
index = [i for i in range(len(image))]
random.shuffle(index) #Permutation aléatoire
index_train = index[0:int(len(index)*0.8)]#Index des données train
index_test = index[int(len(index)*0.8):len(index)]#Index des données test
image_train = [image[i] for i in index_train]
image_test = [image[i] for i in index_test]
label_image_train = [label_image[i] for i in index_train]
label_image_test = [label_image[i] for i in index_test]

X_train = image_train
X_test = image_test
Y_train = label_image_train
Y_test = label_image_test

### Definition de la fonction d'activation sigmoide et sa dérivé
def sigmoide(x):
    return(1/(1+exp(-lambda_0 * x)))

def deriv_sigmoide(x):
    num = lambda_0 * exp(-lambda_0 * x)
    denom = (1+exp(-lambda_0 * x))**2
    return(num/denom)

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))


### Definition de la fonction d'activation ReLU et sa dérivé
def relu(x):
    return(max([0,x]))

def deriv_relu(x):
    if x>0:
        return 1
    else :
        return 0

### Choix des paramètres
n_batch = 1 #on met un batch de 1 pour simplifier
n_epochs = 10000
lambda_0 = 1 #paramètre pour la fonction d'activation sigmoide
alpha = 1e-3 #learning rate
neurones_entrees = len(X_train[1]) # Correspond au nombre de coordonnées du vecteur d'entré
neurones_cachees = 5 # Nombre de couches cachées
n_out = 10 # Nombre de coordonné sortie
poids_initiaux = 0 # Choix des poids initiaux
fonction_activation = sigmoide #Choix de la fonction d'activation
deriv_activation = deriv_sigmoide

### Initialisation des poids
poids_couche_1 = [[poids_initiaux for i in range(neurones_entrees+1)] for i in range(neurones_cachees)]
poids_couche_2 = [[poids_initiaux for i in range(neurones_cachees+1)] for i in range(n_out)]

### Initialisation des poids de manière aléatoire
poids_couche_1 = [[(random.random()-0.5)*2 for i in range(neurones_entrees+1)] for i in range(neurones_cachees)]
poids_couche_2 = [[(random.random()-0.5)*2 for i in range(neurones_cachees+1)] for i in range(n_out)]

### Boucle d'entrainement du réseau
perte_liste = [] #Liste contenat la perte
temps_1 = datetime.datetime.now()
for i in range(n_epochs):
    delta2 = [[0 for j in range(neurones_cachees + 1)] for k in range(n_out)]
    delta1 = [[0 for j in range(neurones_entrees + 1)] for k in range(neurones_cachees)]

    for l in range(n_batch):
        #Début du batch
        id_ent = randint(0, len(X_train) - 1)
        #id_ent = i%(neurones_entrees+1)
        X = X_train[id_ent]

        noeuds = []
        noeuds_in = []
        for k in range(neurones_cachees):
            noeuds_1_in = poids_couche_1[k][neurones_entrees]
            for j in range(neurones_entrees-1):
                noeuds_1_in = noeuds_1_in + X[j] * poids_couche_1[k][j]
            noeuds_1 = fonction_activation(noeuds_1_in)
            noeuds = noeuds + [noeuds_1]
            noeuds_in = noeuds_in + [noeuds_1_in]
        sortie_in = []
        sortie = []
        for k in range(n_out):
            noeuds_2_in = poids_couche_2[k][neurones_cachees]
            for j in range(neurones_cachees):
                noeuds_2_in = noeuds_2_in + noeuds[j] * poids_couche_2[k][j]
            noeuds_2 = fonction_activation(noeuds_2_in)
            sortie_in = sortie_in + [noeuds_2_in]
            sortie = sortie + [noeuds_2]


        # Calcul erreur
        Y_reel = Y_train[id_ent]
        #perte = perte + [CrossEntropy(sortie,Y_reel)]
        perte = 0
        for k in range(n_out):
            perte = perte + 0.5 * (Y_reel[k] - sortie[k])**2
        perte_liste = perte_liste + [perte]

        # Calcul gradient 2ème couche

        for k in range(n_out):
            for j in range(neurones_cachees):
                delta2[k][j] = delta2[k][j] + 2 * (sortie[k] - Y_reel[k]) * deriv_activation(sortie_in[k]) * noeuds[j]
            delta2[k][neurones_cachees] = 2 * (sortie[k] - Y_reel[k]) * deriv_activation(sortie_in[k])

        # Calcul gradient 1ère couche

        delta_in = [0 for i in range(neurones_cachees)]
        for j in range(neurones_cachees):
            for k in range(n_out):
                delta_in[j] = delta_in[j] + 2 * (sortie[k] - Y_reel[k]) * deriv_activation(sortie_in[k])

        for j in range(neurones_cachees):
            for k in range(neurones_entrees):
                delta1[j][k] = 2 *delta_in[j] * deriv_activation(noeuds_in[j]) * X[k]
            delta1[j][neurones_entrees] = 2 * delta_in[j] * deriv_activation(noeuds_in[j])



        # if (i*l%1000 == 0):
        #     print((i*l)/(n_epochs*n_batch))
        #     print(moyenne(perte))


    for l in range(neurones_cachees):
        for j in range(neurones_entrees+1):
            delta1[l][j] = delta1[l][j]/n_batch
    for l in range(n_out):
        for j in range(neurones_cachees + 1):
            delta2[l][j] = delta2[l][j]/n_batch



    # Mise à jour des poids
    for k in range(n_out):
        for j in range(neurones_cachees + 1):
            poids_couche_2[k][j] = poids_couche_2[k][j] - alpha * delta2[k][j]

    for k in range(neurones_cachees):
        for j in range(neurones_entrees + 1):
            poids_couche_1[k][j] = poids_couche_1[k][j] - alpha * delta1[k][j]

    if (i*l%1000 == 0):
        print(i/n_epochs)
        print(moyenne(perte_liste))

temps_2 = datetime.datetime.now()
n_test = len(Y_test)
Y_calcul = [sortie_reseau(X_test[i]) for i in range(n_test)]
print("Le score sur les données test est de :")
print(MSE(Y_calcul,Y_test))
print("Le temps de calcul est :")
print(temps_2 - temps_1)
plt.plot([i for i in range(n_epochs*n_batch)],perte_liste)
plt.show()


### Test sur les données test

n_test = len(Y_test)
Y_calcul = [sortie_reseau(X_test[i]) for i in range(n_test)]


# on prend la sortie et on regarde la valeur qui a la meilleure probabilité
Y_calcul_arr_m1 = []
for i in range(n_test):
    rang = rang_max(Y_calcul[i])
    l = []
    for j in range(n_out):
        if j == rang:
            l = l + [1]
        else :
            l = l + [0]
    Y_calcul_arr_m1 = Y_calcul_arr_m1 + [l]


# Calcul de la matrice de confusion
Y_calcul_arr = Y_calcul_arr_m1
matrice_confusion = [[0 for i in range(n_out)] for i in range(n_out)]
for k in range(len(Y_calcul_arr)):
    i = rang_max(Y_calcul_arr[k])
    j = rang_max(Y_test[k])
    matrice_confusion[i][j] = matrice_confusion[i][j] + 1

score = (matrice_confusion[1][1] + matrice_confusion[0][0])/len(Y_calcul_arr)

affichage_matrice(matrice_confusion)
print(score)

### Test sur des images faites à la main
# Importation du module de lecture d'image
from PIL import Image
# Ouverture de l'image
im = Image.open(r"C:\Users\schoo\OneDrive\Bureau\Mines\Cours\2A\Pro Tech\Code\lettres\1.png")
pixel_map = im.load() #Extraction des pixels
pixel = [] # Convertion des données pour les utiliser
for i in range(28):
    for j in range(28):
        pixel = pixel + [1 - moyenne(pixel_map[i,j])/255]
print(sortie_reseau(pixel))