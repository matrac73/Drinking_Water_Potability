import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # empêcher les warning de s'afficher

## Recupération des données brut sous forme d'un dataframe
df = pd.read_csv(r"./data/drinking_water_potability.csv")
# print(df.shape)
# print(df) # Affichage des données dataframe
# print(df.head()) # tête du dataframe
# print(df.describe()) # statistiques descriptives du dataframe

## Affichage des variables explicatives
data = df.to_numpy() # convertir le dataframe en array
fig = plt.figure(figsize=(15,9)) # création de la fenetre
fig.canvas.set_window_title('Drinking Water Potability - Variable Distribution') # titre de la fenêtre
plt.style.use('seaborn-whitegrid') # style de grille
for i in range(len(data[0][:-1])):
    ax = plt.subplot(3,3,1+i) # changer de subplot
    sns.distplot(df[df.columns[i]], color = '#2a7bff') # afficher le barplot et la repartition
plt.suptitle("Histogrammes des variables explicatives", fontsize=20) # titre principal
plt.show() # afficher le tout

## Check for null values
# print(df.isnull().sum()) # afficher les valeurs manquantes au dataset

## Clean dataset by replacing missing values
df['ph'] = df['ph'].fillna(df['ph'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median()) # remplacer les valeurs manquantes par la moyenne (distribution normale)

## Exploratory Data Analysis

columns = [x for x in df.columns if x != 'Potability'] # selection des variables explicatives
plt.figure(figsize=(15,9)) # création de la fenetre
plt.suptitle("Diagramme moustache en fonction de la potabilité", fontsize=20) # titre principal
for i in range(9):
    plt.subplot(3,3,i+1) # choix du subplot
    sns.boxplot(data=df,x = 'Potability' ,y= columns[i],showfliers=False) # diagrammes moustaches
plt.show() # afficher le tout

plt.figure(figsize=(9,9)) # création de la fenetre
plt.suptitle("Matrice de correlation des variables", fontsize=20) # titre principal
sns.heatmap(df.corr(), annot=True, cmap = 'RdYlGn', vmin=-1, vmax=1) # matrice de corrélation
plt.show() # afficher le tout