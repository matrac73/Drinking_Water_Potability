import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Recupération des données brut sous forme d'un dataframe
df = pd.read_csv(r"./data/drinking_water_potability.csv")

# print(df) # Affichage des données dataframe
# print(df.head()) # tête du dataframe
# print(df.describe()) # statistiques descriptives du dataframe

data = df.to_numpy() # convertir le dataframe en array
# print(data) # Affichage des données array

# import pdb; pdb.set_trace() # debugger

fig = plt.figure(figsize=(15,9)) # création de la fenetre
fig.canvas.set_window_title('Drinking Water Potability - Data Visualization') # titre de la fenêtre
plt.style.use('seaborn-whitegrid') # style de grille
for i in range(len(data[0][:-1])):
    ax = plt.subplot(331+i) # changer de subplot
    ax.set_title(df.columns[i]) # sous-titres
    ax.grid(True) # activer grille
    ax.hist(data[:,i], 60, facecolor='#2ab0ff', edgecolor='#169acf', alpha=0.75) # plot la data
plt.suptitle("Histogrammes des variables explicatives", fontsize=20) # titre principal
plt.show() # afficher le tout