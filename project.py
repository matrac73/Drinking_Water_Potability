import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Recupération des données brut sous forme d'un dataframe
df = pd.read_csv(r"./data/drinking_water_potability.csv")

# print(df) # Affichage des données
print(df.head()) # tête du dataframe

print(df.describe()) # statistiques descriptives du dataframe