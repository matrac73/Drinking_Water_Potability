import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import scipy
import statsmodels.api as sm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import time as t

def avoid_warnings():
    print("Supression des Warnings...", end="\r")
    warnings.filterwarnings("ignore", category=FutureWarning) # empêcher les warning de s'afficher
    print(f"{t.ctime(t.time())} : Warnings Suprimés")

def load_data(csv_path, disp):
    """Recupération des données brut sous forme d'un dataframe"""
    print("Chargement des données...", end="\r")
    global df
    df = pd.read_csv(csv_path)
    print(f"{t.ctime(t.time())} : Données chargés")
    if disp:
        print(f"Dimentions du dataframe : {df.shape}") # Affichage des dimensions de df
        print(f"Dataframe Brut : \n{df}") # Affichage des données dataframe
        print(f"Tête du dataframe : \n{df.head()}") # tête du dataframe
        print(f"Satistiques descriptives du Dataframe : \n{df.describe()}") # statistiques descriptives du dataframe
    
def display_explanatory_variables(disp):
    """Affichage des variables explicatives"""
    if disp:
        data = df.to_numpy() # convertir le dataframe en array
        fig = plt.figure(figsize=(15,9)) # création de la fenetre
        fig.canvas.set_window_title('Drinking Water Potability - Variable Distribution') # titre de la fenêtre
        plt.style.use('seaborn-whitegrid') # style de grille
        for i in range(len(data[0][:-1])):
            ax = plt.subplot(3,3,1+i) # changer de subplot
            sns.distplot(df[df.columns[i]], color = '#2a7bff') # afficher le barplot et la repartition
        plt.suptitle("Histogrammes des variables explicatives", fontsize=20) # titre principal
        plt.show() # afficher le tout

def check_null_values(disp):
    """Check for null values in the dataframe"""
    print("Recherche de données nulles...", end="\r")
    dfn = df.isnull()
    print(f"{t.ctime(t.time())} : Données nulles trouvées")
    if disp:
        print(dfn.sum()) # afficher les valeurs manquantes au dataset

def cleaning_dataset():
    """Clean dataset by replacing missing values"""
    print("Nettoyage des données nulles...", end="\r")
    df['ph'] = df['ph'].fillna(df['ph'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
    print(f"{t.ctime(t.time())} : Données nulles remplacées")

def delete_outliers():
    """Clean dataset by deleting outliers"""
    global df
    print("Suppression des données aberrantes...", end="\r")
    z_scores = scipy.stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    print(f"{t.ctime(t.time())} : Données aberrantes remplacées")


def boxplot():
    """Diagramme moustache"""
    columns = [x for x in df.columns if x != 'Potability'] # selection des variables explicatives
    plt.figure(figsize=(15,9)) # création de la fenetre
    plt.suptitle("Diagramme moustache en fonction de la potabilité", fontsize=20) # titre principal
    for i in range(9):
        plt.subplot(3,3,i+1) # choix du subplot
        sns.boxplot(data=df,x = 'Potability' ,y= columns[i],showfliers=False) # diagrammes moustaches
    plt.show() # afficher le tout

def heatmap_corr():
    plt.figure(figsize=(9,9)) # création de la fenetre
    plt.suptitle("Matrice de correlation des variables", fontsize=20) # titre principal
    sns.heatmap(df.corr(), annot=True, cmap = 'RdYlGn', vmin=-1, vmax=1) # matrice de corrélation
    plt.show() # afficher le tout

# ## Séparer les données en train set et test set
# X_train, X_test, y_train, y_test = train_test_split(df[[c for c in df.columns if c != 'Potability']],df['Potability'],train_size = 0.7,random_state = 1)

# ## Scaling des données des variables explicatives
# sc = StandardScaler()
# X_train[X_train.columns] = sc.fit_transform(X_train)
# X_test[X_test.columns] = sc.transform(X_test)
# X_train, y_train = SMOTE(random_state=1,n_jobs=-1).fit_resample(X_train,y_train)
# X_train_sm = sm.add_constant(X_train)
# lm = sm.GLM(y_train,X_train_sm,family=sm.families.Binomial()).fit()
# print(lm.summary())

# def vif(data):
#     res = pd.DataFrame()
#     res['Feature'] = data.columns
#     res['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
#     return res
# print(vif(X_train_sm))

# X_test_sm = sm.add_constant(X_test)
# y_train_pred = lm.predict(X_train_sm)

# def tp_fp(cf):
#     fp = cf[0,1]/(cf[0,0] + cf[0,1])
#     tp = cf[1,1]/(cf[1,0] + cf[1,1])
#     return fp,tp


# def plot_roc(data,truth):
#     cutoff = [0.001*i for i in range(1,1000)]
#     x = []
#     y = []
#     for c in cutoff:
#         #print(data)
#         data_temp = data.apply(lambda x: 1 if x>=c else 0)
#         cf = confusion_matrix(truth,data_temp)
#         x.append(tp_fp(cf)[0])
#         y.append(tp_fp(cf)[1])
#     plt.plot(x,y)
#     plt.show()

# plot_roc(y_train_pred,y_train)

# fp,tp,_ = roc_curve(y_train,y_train_pred)
# plt.plot(fp,tp)

# y_test_pred = lm.predict(X_test_sm).apply(lambda x: 1 if x>= 0.5 else 0)

# cf = confusion_matrix(y_test,y_test_pred)
# acc = (cf[0,0] + cf[1,1])/(cf[0,0] + cf[0,1] + cf[1,1] +cf[1,0])
# tpr = cf[1,1]/(cf[1,0] + cf[1,1])
# print(f"tpr = {tpr} accuracy = {acc}")

# dt = DecisionTreeClassifier(random_state=1)
# params = {
#     "min_samples_split": [10,20,100],
#     "max_depth": [5,10,50],
#     "min_samples_leaf": [10,20,50],
#     "max_leaf_nodes": [10,20,100]
# }

# dt_grid = GridSearchCV(estimator=dt,param_grid=params,cv=5,scoring='balanced_accuracy',verbose=10,n_jobs = -1).fit(X_train,y_train)

# # gb = cv.best_estimator_

# # y_train_pred = gb.predict(X_train)
# # cf = confusion_matrix(y_train,y_train_pred)

# # print(cf)
# # acc = (cf[0,0] + cf[1,1])/np.sum(cf)
# # recall = (cf[1,1])/(cf[1,1] + cf[1,0])
# # spec = (cf[0,0])/(cf[0,1] + cf[0,0])
# # print(f"Accuracy = {acc} Recall = {recall} Specificity = {spec}")
