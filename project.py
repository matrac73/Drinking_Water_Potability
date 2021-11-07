import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import warnings
import time as t
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats

def avoid_warnings():
    print("Supression des Warnings...", end="\r")
    warnings.filterwarnings("ignore", category=FutureWarning) # empêcher les warning de s'afficher
    warnings.filterwarnings("ignore", category=UserWarning) # empêcher les warning de s'afficher
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

def cleaning_dataset(mode = 'mean'):
    """Clean dataset by replacing or deleting missing values"""
    print("Nettoyage des données nulles...", end="\r")
    if mode == "mean":
        df['ph'] = df['ph'].fillna(df['ph'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
        df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median()) # remplacer les valeurs manquantes par la moyenne (distribution normale)
        print(f"{t.ctime(t.time())} : Données nulles remplacées")
    elif mode == "delete":
        df.dropna(inplace=True)
        print(f"{t.ctime(t.time())} : Données nulles supprimées")
    

def cope_outliers(mode = 'Q+1.5'):
    """Clean dataset by deleting outliers"""
    global df
    print("Modification des données aberrantes...", end="\r")
    
    columns = [x for x in df.columns if x != 'Potability']
    Q1, Q3 = df.quantile(.25), df.quantile(.75)
    IQR = Q3 - Q1
    if mode == "Q+1.5":
        for k in columns :
            df.loc[df[k] >= Q3[k]+1.5*IQR[k], k] = Q3[k]+1.5*IQR[k]
            df.loc[df[k] <= Q1[k]-1.5*IQR[k], k] = Q1[k]-1.5*IQR[k]
    elif mode == "Q":
        for k in columns :
            df.loc[df[k] >= Q3[k], k] = Q3[k]
            df.loc[df[k] <= Q1[k], k] = Q1[k]
    elif mode == "delete":
        for k in columns :
            df.drop(list(df.loc[df[k] >= Q3[k]+1.5*IQR[k], k].index))
            df.drop(list(df.loc[df[k] <= Q1[k]-1.5*IQR[k], k].index))
    print(f"{t.ctime(t.time())} : Données aberrantes gérées")

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

def split_dataset(ratio, disp):
    """Séparer les données en train set et test set"""
    print("Séparation du dataset...", end="\r")
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(df[[c for c in df.columns if c !='Potability']],df['Potability'],train_size = ratio,random_state = 1)
    print(f"{t.ctime(t.time())} : Dataset séparé")
    if disp:
        print(f"X_train : {X_train.shape}")
        print(f"X_test  : {X_test.shape}")
        print(f"y_train : {y_train.shape}")
        print(f"y_test  : {y_test.shape}")

def scaling_trainset():
    """Scaling des données des variables explicatives"""
    print("Mise à l'echelle du trainset...", end="\r")
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)
    print(f"{t.ctime(t.time())} : Trainset mis à l'echelle")

def set_metric(metric_score):
    global metric, metric_name
    if metric_score == 'f1_score':
        metric = f1_score
        metric_name = 'F1_score'
    elif metric_score == 'accuracy':
        metric = accuracy_score
        metric_name = 'Accuracy'
    else:
        print('Metric score not valid, using accuracy')
        metric = accuracy_score
        metric_name = 'Accuracy'
        
def fitting_KNN_model():
    """Création d'un model K-Nearest Neighbours"""
    global KNN
    KNN = KNeighborsClassifier()
    return fitting_model(KNN, "KNN")
    
def fitting_LR_model():
    """Création d'un model de regression logistique"""
    global LR
    LR = LogisticRegression()
    return fitting_model(LR, "LR")
    
def fitting_RF_model():
    """Création d'un model Random Forest"""
    global RF
    RF = RandomForestClassifier()
    return fitting_model(RF, "RF")
    
def fitting_SVM_model():
    """Création d'un model Support Vector Machine"""
    global SVM
    SVM = SVC()
    return fitting_model(SVM, "SVM")

def fitting_XGboost_model():
    """Création d'un model XGboost"""
    global XGboost
    XGboost = XGBClassifier()
    return fitting_model(XGboost, "XGboost")

def fitting_model(model, model_name):
    """Création d'un model"""
    print(f"Ajustement du model { model_name }...", end="\r")
    model.fit(X_train, y_train)
    print(f"{t.ctime(t.time())} : model { model_name } ajusté")

def testing_KNN_model():
    """Test du model K-Nearest Neighbours"""
    print("Ajustement du model KNN...", end="\r")
    return testing_model(KNN, "KNN")
    
def testing_LR_model():
    """Test du model de regression logistique"""
    return testing_model(LR, "LR")
    
def testing_RF_model():
    """Test du model Random Forset"""
    return testing_model(RF, "RF")
    
def testing_SVM_model():
    """Test du model Support Vector Machine"""
    return testing_model(SVM, "SVM")

def testing_XGboost_model():
    """Test du model XGboost"""
    return testing_model(XGboost, "XGboost")

def testing_model(model, model_name):
    """Test du model"""
    global metric, metric_name
    print(f"Ajustement du model {model_name}...", end="\r")
    y_test_hat = model.predict(X_test)
    accuracy = round(metric(y_test, y_test_hat)*100, 2)
    print(f"{t.ctime(t.time())} : model {model_name} testé")
    print(f"{metric_name} {model_name} : {accuracy} %\n")

def tuning_kNN_hyperparameters(param_grid, method):
    global KNN
    return tuning_hyperparameters(KNN, param_grid, method)

def tuning_LR_hyperparameters(param_grid, method):
    global LR
    return tuning_hyperparameters(LR, param_grid, method)

def tuning_RF_hyperparameters(param_grid, method):
    global RF
    return tuning_hyperparameters(RF, param_grid, method)

def tuning_SVM_hyperparameters(param_grid, method):
    global SVM
    return tuning_hyperparameters(SVM, param_grid, method)

def tuning_XGboost_hyperparameters(param_grid, method):
    global XGboost
    return tuning_hyperparameters(XGboost, param_grid, method)

def tuning_hyperparameters(estimator, param_grid, method):
    print(f"Recherche des meilleurs hyperparamètres par méthode { method } ...", end="\r")
    start_time = t.time()
    if method == "RandomizedSearchCV":
        model = RandomizedSearchCV(estimator = estimator, param_distributions = param_grid, n_iter = 100, cv = 5, verbose=0, random_state=1984, n_jobs = -1)
    if method == "GridSearchCV":
        model = GridSearchCV(estimator, param_grid, cv = 5, verbose=1, n_jobs = 1)
    model.fit(X_train, y_train)
    print(f"{t.ctime(t.time())} : Meilleurs hyperparamètres trouvés                ")
    print(f"Durée de la recherche : {round(t.time()-start_time, 2)} secondes")
    print(f"Meilleurs Hyperparamètres par méthode { method } : {model.best_params_}")
    return model.best_params_

def fitting_kNN_tuned_model(dict):
    kNN = KNeighborsClassifier(n_neighbors=dict["n_neighbors"], 
                               weights=dict["weights"], 
                               leaf_size=dict["leaf_size"], 
                               p=dict["p"])
    return fitting_tuned_model(kNN, "kNN")

def fitting_LR_tuned_model(dict):
    LR = LogisticRegression(C=dict['C'], 
                            penalty=dict['penalty'], 
                            solver=dict['solver'])
    return fitting_tuned_model(LR, "LR")

def fitting_RF_tuned_model(dict):
    RF = RandomForestClassifier(n_estimators=dict["n_estimators"], 
                                min_samples_split=dict["min_samples_split"], 
                                min_samples_leaf=dict["min_samples_leaf"], 
                                max_depth=dict["max_depth"], 
                                bootstrap=dict["bootstrap"])
    return fitting_tuned_model(RF, "RF")

def fitting_SVM_tuned_model(dict):
    SVM = SVC(kernel = dict['kernel'], 
              C = dict['C'], 
              gamma = dict['gamma'])
    return fitting_tuned_model(SVM, "SVM")

def fitting_XGboost_tuned_model(dict):
    XGBoost = XGBClassifier(gamma = dict['gamma'], 
                            max_depth = dict['max_depth'], 
                            min_child_weight = dict["min_child_weight"],  
                            subsample = dict["subsample"], 
                            colsample_bytree = dict["colsample_bytree"])
    return fitting_tuned_model(XGBoost, "XGBoost")

def fitting_tuned_model(model, model_name):
    global metric, metric_name
    print(f"Ajustement et test du meilleur model {model_name} ...", end="\r")
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    accuracy = round(metric(y_test, y_test_hat)*100, 2)
    print(f"{t.ctime(t.time())} : meilleur model {model_name} testé et ajusté")
    print(f"{metric_name} {model_name} : {accuracy} %\n")
