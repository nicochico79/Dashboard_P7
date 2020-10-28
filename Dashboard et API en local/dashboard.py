import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import requests
import shap
import matplotlib.pyplot as plt



def main():

    API_URL = 'http://127.0.0.1:5000/'

    # 1. Chargement des données via l'API
    
    @st.cache(allow_output_mutation=True)
    def load_data(url):
    	req = requests.get(url)
    	content = json.loads(req.content.decode('utf-8'))
    	return pd.DataFrame(content['data'])

    data_load_state = st.text('Loading data...')
    df_desc = load_data(API_URL + 'info')
    df_desc = df_desc[['AGE', 'GENDER',
                    'FAMILY STATUS', 'FAMILY MEMBERS', 'NUMBER CHILDREN',
                    'FLAT OWNER', 'EDUCATION TYPE', 'SOURCE OF INCOME',
                    'OCCUPATION TYPE','YEARS EMPLOYED',
                    'ANNUAL INCOME', 'AMOUNT CREDIT',
                    'AMOUNT ANNUITY', 'GOODS VALUE']]

    averages = pd.read_csv('./data/moyennes.csv')
    averages['AGE'] = (averages['DAYS_BIRTH']/-365).astype(int)
    averages['YEARS EMPLOYED'] = round((averages['DAYS_EMPLOYED']/-365), 0)
    averages.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    averages = averages.rename(columns={'CODE_GENDER': "GENDER", "NAME_FAMILY_STATUS": "FAMILY STATUS",
                             'CNT_FAM_MEMBERS': 'FAMILY MEMBERS', 'FLAG_OWN_REALTY': "FLAT OWNER",
                             'NAME_EDUCATION_TYPE': 'EDUCATION TYPE', 'NAME_INCOME_TYPE': 'SOURCE OF INCOME',
                             'OCCUPATION_TYPE': 'OCCUPATION TYPE', 'AMT_INCOME_TOTAL': "ANNUAL INCOME",
                             'AMT_CREDIT': 'AMOUNT CREDIT', 'AMT_ANNUITY': 'AMOUNT ANNUITY',
                             'AMT_GOODS_PRICE': 'GOODS VALUE', 'CNT_CHILDREN': 'NUMBER CHILDREN'})
    
    data_load_state.text('')

    

    # 2. Récupération des informations générales
    
    st.title('Dashboard Projet 7')

    # 2.1 Selection d'1 client
    id_client = st.sidebar.selectbox('Select ID Client :', df_desc.index)

    # 2.2 Affichage de ses informations générales dans la barre latérale
    st.sidebar.table(df_desc.loc[id_client][:9])



    # 3. Positionnement du client par rapport à clients défauts et clients solvables - données quant

    donnees = ['AGE', 'YEARS EMPLOYED', 'FAMILY MEMBERS','NUMBER CHILDREN','ANNUAL INCOME', 'AMOUNT ANNUITY',
               'AMOUNT CREDIT', 'GOODS VALUE']

    # 3.1 Chargement des données client
    url_data_client = API_URL + 'info/' + str(id_client)
    req = requests.get(url_data_client)
    content = json.loads(req.content.decode('utf-8'))
    data_client = pd.DataFrame(content['data']).copy()

    # 3.2 Comparaison données clients et données moyennes
    st.header('1. Comparaison données client versus autres clients')
    df_pos = pd.concat([averages, data_client], join = 'inner').round(2)

    fig_1 = make_subplots(rows=4, cols=2,
                         subplot_titles=[donnees[i] for i in range(8)],
                          shared_xaxes = True,
                          vertical_spacing = 0.05
                          )

        # Ajout des graphes
    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[0]].iloc[i] for i in range(3)],
                              name=donnees[0]
                                ),
                   row=1, col=1, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Client défault", "Client"],
                             y=[df_pos[donnees[1]].iloc[i] for i in range(3)],
                              name=donnees[1]
                                ),
                   row=1, col=2, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[2]].iloc[i] for i in range(3)],
                              name=donnees[2]
                                ),
                   row=2, col=1, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[3]].iloc[i] for i in range(3)],
                              name=donnees[3]
                                ),
                   row=2, col=2, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[4]].iloc[i] for i in range(3)],
                                marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[4]
                                ),
                   row=3, col=1, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[5]].iloc[i] for i in range(3)],
                            marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[5]
                            ),
                   row=3, col=2, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[6]].iloc[i] for i in range(3)],
                                marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[6]
                                ),
                   row=4, col=1, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Client défault", "Client"],
                              y=[df_pos[donnees[7]].iloc[i] for i in range(3)],
                                marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[7]
                                ),
                   row=4, col=2, secondary_y=False)
    fig_1['layout'].update(height = 800, width = 600, showlegend = False)

    for i in fig_1['layout']['annotations']:
        i['font'] = dict(size=12)
    
    st.plotly_chart(fig_1)



    # 4. Prédictions

    # 4.1 Chargement des données client et récupération de sa probabilité de défaut 
    url_data_client = API_URL + 'processed/' + str(id_client)
    req = requests.get(url_data_client)
    content = json.loads(req.content.decode('utf-8'))
    prediction_client = content['prediction']

    # 4.2 Même opération pour ses voisins
    url_voisins_client = API_URL + 'voisins/' + str(id_client)
    req = requests.get(url_voisins_client)
    content = json.loads(req.content.decode('utf-8'))
    prediction_voisins = content['prediction']

    # 4.3 Représentation graphique sous forme de jauge horizontale
    if st.sidebar.checkbox("Afficher Proba Défaut", False, key = 2):
        st.header('2. Probabilité de défaut pour le client {}'.format(id_client))
        fig0 = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = prediction_client,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': prediction_voisins,
                 'increasing': {'color': 'red'},
                'decreasing' : {'color' : 'green'},
                 'position': "top"},
        title = {'text':"<b>En %</b><br><span style='color: gray; font-size:0.8em'></span>", 'font': {"size": 14}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100]},
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75, 'value': prediction_voisins},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}],
            'bar': {'color': "darkblue"}}))
        fig0.update_layout(height = 250)

        st.plotly_chart(fig0)

        st.markdown('Pour le client sélectionné : **{0:.1f}%**'.format(prediction_client))
        st.markdown('Pour les clients similaires : **{0:.1f}%** (critères de similarité : âge, genre,\
         situation familiale, éducation, profession)'.format(prediction_voisins))



    # 5. Interprétation des résultats

    features = ['EXT_SOURCE_2' ,'EXT_SOURCE_3','TENOR','LTV','EXT_SOURCE_1','Life_employed','AMOUNT ANNUITY']
    
    feature_desc = {'EXT_SOURCE_2' : 'Score normalisé attribué par organisme indépendant 2',
                    'EXT_SOURCE_3' :  'Score normalisé attribué par organisme indépendant 3',
                     'TENOR': 'Durée du crédit demandé',
                     'LTV': 'Valeur du crédit par rapport au bien à financer',
                     'EXT_SOURCE_1' :  'Score normalisé attribué par organisme indépendant 1',
                     'Life_employed' : 'Années travaillées en pourcentage',
                    'AMOUNT ANNUITY' : 'Montant des annuités'}

    # 5.1 Chargement des données client
    req = requests.get(url_data_client)
    content = json.loads(req.content.decode('utf-8'))
    data_client = pd.DataFrame(content['data']).copy()

    # 5.2 Chargement des moyennes des voisins
    req = requests.get(url_voisins_client)
    content = json.loads(req.content.decode('utf-8'))
    mean_vois = pd.DataFrame(content['mean']).copy()

    # 5.3 Aggrégation des données dans un dataframe (une ligne par type de client)

    data_client = data_client.rename(columns={'AMT_ANNUITY': 'AMOUNT ANNUITY'})

    mean_vois = mean_vois.rename(columns={'AMT_ANNUITY': 'AMOUNT ANNUITY'})

    dfcomp = pd.concat([averages, mean_vois, data_client], join = 'inner').round(2)

    if st.sidebar.checkbox("Afficher Détails Modèle", False, key = 0):
        st.header('3. Explication origine probabilité défaut')
        feature = st.selectbox('Selectionnez la variable à analyser:', features)
        fig3 = go.Figure(data=[go.Bar(x=dfcomp[feature],
                                    y=['Moyenne clients solvables ',
                                          'Moyenne clients défaut ',
                                          'Moyenne clients similaires ',
                                          'Client Sélectionné '],
                                       marker_color=['green','red', 'blue', 'lightblue'],
                                       orientation ='h')])

        fig3.update_layout(title_text=feature_desc[feature])

        st.plotly_chart(fig3)




if __name__== '__main__':
    main()




