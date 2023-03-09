"""=
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DH0yFSy9FGo58MSlo76GFHRPTgj2I9wE

Le réseaux de neurones de convolution appliqué aux données MNIST
"""

# Pour commencer, nous devons importer quelques bibliothèques de PyTorch
import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests
from PIL import Image
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy  as np

#Définir la taille du batch size à 64

numb_batch=64

"""
Nous devons transformer l'image en un tenseur qui peut être utilisé, donc nous faisons torchvision.transforms.ToTensor().

Nous obtenons les données d'entraînement de la bibliothèque MNIST et définissons le téléchargement à True.
Ensuite, nous avons besoin de transformer les images.
La même chose peut être faite pour les données de validation sauf que train est False.

Nous avons également besoin des chargeurs de données pour chaque jeu de données et définissons la taille du batch size au nombre souhaité, 64.
"""

T=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data=torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=T)
val_data=torchvision.datasets.MNIST('mnist_data',train=False,download=True,transform=T)
train_dl=torch.utils.data.DataLoader(train_data,batch_size=numb_batch)
val_dl=torch.utils.data.DataLoader(val_data,batch_size=numb_batch)

print(train_data.data.size())
print(val_data.data.size())

# Le modèle de convolution
def creat_lenet():
  model=nn.Sequential(
      # Première couche
      nn.Conv2d(1,6,5,padding=2),
      nn.LeakyReLU(),
      nn.AvgPool2d(2,stride=2),
      # Deuxième couche
      nn.Conv2d(6,16,5,padding=0),
      nn.LeakyReLU(),
      nn.AvgPool2d(2,stride=2),
      # Flatten
      nn.Flatten(),
      nn.Linear(400, 120),
      nn.LeakyReLU(),
      nn.Linear(120, 84),
      nn.Tanh(),
      nn.Linear(84, 10)
  )
  return model

"""
Cette validation utilisera l'ensemble de validation des chiffres manuscrits et calculera combien d'images sont prédites correctement sur le nombre total d'images.
Il s'agit juste d'une simple boucle à travers chaque image dans le dataloader de validation.
"""

def validate(model,data):
  total=0
  correct=0
  for i, (images,labels) in enumerate(data):
    images=images.cuda()
    x=model(images)
    value,pred=torch.max(x,1)
    pred=pred.data.cpu()
    total+=x.size(0)
    correct+=torch.sum(pred==labels)
  return correct*100./total

"""
Pour l'entraînement, nous définirons la valeur par défaut à 10 époques, le taux d'apprentissage à 0.001, et le dispositif au cpu de la machine.
"""

def train(numb_epoch=3,lr=1e-3,device="cpu"):
  accuracies=[]
  cnn=creat_lenet().to(device)
  # La fonction de perte
  cec=nn.CrossEntropyLoss()
  # La fonction d'optimisation
  optimizer=optim.RMSprop(cnn.parameters(),lr=lr)
  max_accuracy=0
  for epoch in range(numb_epoch):
    for i, (images,labels) in enumerate(train_dl):
      images=images.to(device)
      labels=labels.to(device)
      optimizer.zero_grad()
      pred=cnn(images)
      loss=cec(pred,labels)
      loss.backward()
      optimizer.step()
    accuracy=float(validate(cnn,val_dl))
    accuracies.append(accuracy)
    if accuracy>max_accuracy:
      best_model=copy.deepcopy(cnn)
      max_accuracy=accuracy
      print("Saving Best Model with Accuracy :",accuracy)
    print('Epoch:',epoch+1,"Accuracy:",accuracy,'%')
  plt.plot(accuracies)
  return best_model

"""
Maintenant, nous vérifions si un GPU est disponible. Si c'est le cas, nous pouvons l'utiliser.
Sinon, nous devons utiliser le CPU.
"""

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")
device

"""
Nous allons maintenant appeler la fonction de formation pour former réellement le modèle.
Puisque la fonction renvoie le meilleur modèle, nous le stockons dans le nom lenet.
J'ai choisi d'appeler la fonction avec 10 époques mais vous pouvez essayer différentes valeurs et voir ce qui fonctionne le mieux.
Puisque j'ai un GPU disponible à utiliser, je dois régler le paramètre sur le GPU.
Si vous n'avez pas de GPU disponible, vous pouvez laisser de côté le device=device car par défaut la fonction utilisera le CPU.
"""

lenet=train(10,1e-3, device=device)

"""
Bien que nous ayons déjà utilisé les données de validation pour déterminer la précision de notre modèle, nous voulons également voir où le modèle s'est embrouillé.
Pour ce faire, nous disposerons d'une liste de prédictions et de vérité terrain pour chaque image de l'ensemble de validation.
"""

def predict_dl(model,data):
  y_pred=[]
  y_true=[]
  for i,(images,labels) in enumerate(data):
    images=images.cuda()
    x=model(images)
    value,pred=torch.max(x,1)
    pred=pred.data.cpu()
    y_pred.extend(list(pred.numpy()))
    y_true.extend(list(labels.numpy()))
  return np.array(y_pred),np.array(y_true)

#Les predictions
y_pred, y_true = predict_dl(lenet, val_dl)

# calcul de RMSE
rmse=np.sqrt(np.mean((y_pred-y_true)**2))
print(rmse)

# La matrice de confusion
pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0,10)))

"""
Maintenant, nous pouvons obtenir n'importe quelle image sur Internet avec un seul chiffre et voir ce que le modèle prédit.
Nous utilisons la bibliothèque requests pour obtenir le chemin d'accès, puis nous pouvons accéder au contenu du chemin.
Ensuite, nous devons la redimensionner à (28, 28) afin qu'elle soit accessible par le modèle.

Enfin, nous appelons le modèle avec l'image transformée (de l'image PIL au tenseur).
Nous devons envoyer une image sous forme de flottant et la définir sur le dispositif que nous voulons utiliser.
Après que le modèle ait prédit et retourné une sortie, nous devons faire un softmax afin de normaliser la sortie en probabilité (0.0 à 1.0).
"""

from io import BytesIO
def inference(path, model, device):
    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert(mode="L")
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1))/255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()

# importer une image de l'internet
path = "https://previews.123rf.com/images/aroas/aroas1704/aroas170400068/79321959-handwritten-sketch-black-number-8-on-white-background.jpg"
r = requests.get(path)
with BytesIO(r.content) as f:
    img = Image.open(f).convert(mode="L")
    img = img.resize((28, 28))
x = (255 - np.expand_dims(np.array(img), -1))/255.

#Afficher l'image
plt.imshow(x.squeeze(-1), cmap="gray")

# Prediction
pred = inference(path, lenet, device=device)
pred_idx = np.argmax(pred)
print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")