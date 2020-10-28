import flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import json
import pickle


app = flask.Flask(__name__)
app.config["DEBUG"] = True


# 1. PREPARATION DES DONNEES

# 1.1 Chargement des données descriptives clients dataframe TEST et prétraitement passage en années

n_rows=500 # On limite les données à 500 clients

data_test = pd.read_csv('./data/data_test_clean.csv', nrows=n_rows, index_col='SK_ID_CURR')

cols_keep = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
             'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
             'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
             'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE',
             'CNT_FAM_MEMBERS','CNT_CHILDREN', 'FLAG_OWN_REALTY',
             'NAME_INCOME_TYPE']
df_desc = data_test[cols_keep]

df_desc['AGE'] = (df_desc['DAYS_BIRTH']/-365).astype(int)
df_desc['YEARS EMPLOYED'] = round((df_desc['DAYS_EMPLOYED']/-365), 0)
df_desc.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

df_desc = df_desc[[ 'AGE', 'CODE_GENDER',
                    'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN',
                    'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE',
                    'NAME_INCOME_TYPE', 'OCCUPATION_TYPE','YEARS EMPLOYED',
                    'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                    'AMT_ANNUITY', 'AMT_GOODS_PRICE']]

df_desc.columns = ['AGE', 'GENDER',
                    'FAMILY STATUS', 'FAMILY MEMBERS', 'NUMBER CHILDREN',
                    'FLAT OWNER', 'EDUCATION TYPE', 'SOURCE OF INCOME',
                    'OCCUPATION TYPE','YEARS EMPLOYED',
                    'ANNUAL INCOME', 'AMOUNT CREDIT',
                    'AMOUNT ANNUITY', 'GOODS VALUE']


# 1.2 Chargement des données prétraitées

data_clean = pd.read_csv('./data/data_clean.csv', nrows=n_rows, index_col=0)
data_clean = data_clean.drop(['TARGET',
                             'AMT_CREDIT','REGION_RATING_CLIENT',
                              'OBS_60_CNT_SOCIAL_CIRCLE','DAYS_EMPLOYED']
                             , axis=1)


# 1.3 Définition de la fonction Voisins (afin de positionner client
# par rapport à groupe de voisins définis par distance selon 10 critères communs)

from sklearn.neighbors import KDTree
df_voisins = pd.get_dummies(df_desc.iloc[:,:9])
tree = KDTree(df_voisins)


# 1.4 Chargement du modèle LGBM pré-entraîné
with open('./data/light_gbm.pickle', 'rb') as file : 
	model = pickle.load(file)



# 2. PROGRAMMATION DE L'API

# 2.1 Page d'accueil
@app.route("/")
def hello():
    return "API Tableau de Bord \'Projet 7\' "


# 2.2 Page Informations Générales
@app.route('/info', methods=['GET'])
def get_infos():
		
    # Chargement des données en format JSON
    info_json = json.loads(df_desc.to_json())
    return jsonify({ 'data' : info_json})	


# 2.3 Page Informations Générales pour un client donné
@app.route('/info/<int:client_id>', methods=['GET'])
def get_info_id(client_id):
	
    info_client_select = df_desc.loc[client_id:client_id]
    info_client_json = json.loads(info_client_select.to_json())
    return jsonify({ 'data' : info_client_json})


# 2.4 Récupération des données pré-traitées et prédiction de défaut pour un client donné
@app.route('/processed/<int:client_id>', methods=['GET'])
def get_data_pred(client_id):
	
    data_client_select = data_clean.loc[client_id:client_id]
    data_client_json = json.loads(data_client_select.to_json())

    # Prédiction avec modèle pré-entraîné
    client_pred = 100*model.predict_proba(data_client_select)[0][1]

    return jsonify({ 'data' : data_client_json,
                     'prediction' : client_pred})


# 2.5 Prédiction moyenne sur 10 voisins les plus proches
@app.route('/voisins/<int:client_id>', methods=['GET'])
def voisins(client_id):
    # Récupération des indices des 10 plus proches voisins
    idx_voisins = tree.query(df_voisins.loc[client_id:client_id].fillna(0), k=10)[1][0]
    # Récupération des données 
    data_voisins = data_clean.iloc[idx_voisins]
    # Prédictions
    predict_voisins = 100*model.predict_proba(data_voisins).mean(axis=0)[1]
    # Moyennes des features des voisins
    mean_voisins = pd.DataFrame(data_voisins.mean(), columns=['voisins']).T
    #  Conversion en JSON
    mean_vois_json = json.loads(mean_voisins.to_json())

    return jsonify({ 'mean' : mean_vois_json,
			'prediction' : predict_voisins})    


app.run()
