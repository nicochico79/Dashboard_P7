import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.neighbors import KDTree
import lightgbm
import pickle


def main():

    # 1. Chargement des données et préparation

    # On commence par réduire la taille des dataframes
    n_rows = 500
    
       
    # 1.1 Données informations générales client

    @st.cache(allow_output_mutation=True)
    def infos_gals(): 
        data_test = pd.read_csv('./Data/data_test_clean.csv', nrows=n_rows, index_col='SK_ID_CURR')

        cols_keep = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
             'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
             'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
             'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE',
             'CNT_FAM_MEMBERS','CNT_CHILDREN', 'FLAG_OWN_REALTY',
             'NAME_INCOME_TYPE']

        df_desc = data_test[cols_keep]

        df_desc['AGE'] = (df_desc['DAYS_BIRTH']/-365).astype(int)
        df_desc['YEARS EMPLOYED'] = round((df_desc['DAYS_EMPLOYED']/-365), 0) #Je garde les valeurs négatives pour les retraités et les personnes au chômage comme dans le modèle
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
        return df_desc
        

    # 1.2 Données prétraitées

    @st.cache(allow_output_mutation=True)
    def data_processed(): 
        data_pross = pd.read_csv('./Data/data_clean.csv', nrows=n_rows, index_col=0)
        data_pross = data_pross.drop(['TARGET',
                             'AMT_CREDIT','REGION_RATING_CLIENT',
                              'OBS_60_CNT_SOCIAL_CIRCLE','DAYS_EMPLOYED']
                             , axis=1)
        return data_pross

    data_load_state = st.text('Chargement données...')

    df_desc = infos_gals()
    data_clean = data_processed()


    # 1.3 Données moyennes clients défault vs payeurs

    averages = pd.read_csv('./Data/moyennes.csv')
    averages['AGE'] = (averages['DAYS_BIRTH']/-365).astype(int)
    averages['YEARS EMPLOYED'] = round((averages['DAYS_EMPLOYED']/-365), 0)
    averages.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    averages = averages.rename(columns={'CODE_GENDER': "GENDER", "NAME_FAMILY_STATUS": "FAMILY STATUS",
                             'CNT_FAM_MEMBERS': 'FAMILY MEMBERS', 'FLAG_OWN_REALTY': "FLAT OWNER",
                             'NAME_EDUCATION_TYPE': 'EDUCATION TYPE', 'NAME_INCOME_TYPE': 'SOURCE OF INCOME',
                             'OCCUPATION_TYPE': 'OCCUPATION TYPE', 'AMT_INCOME_TOTAL': "ANNUAL INCOME",
                             'AMT_CREDIT': 'AMOUNT CREDIT', 'AMT_ANNUITY': 'AMOUNT ANNUITY',
                             'AMT_GOODS_PRICE': 'GOODS VALUE', 'CNT_CHILDREN': 'NUMBER CHILDREN'})

    # 1.4 Chargement du modèle LGBM pré-entraîné

    with open('./Data/model.pickle', 'rb') as file :
        model = pickle.load(file)


    # 1.5 Dataframe Voisins (afin de positionner client par rapport à groupe de clients similaires selon 10 critères communs)

    df_voisins = pd.get_dummies(df_desc.iloc[:,:9])
    tree = KDTree(df_voisins)
 
    data_load_state.text('')

    

    # 2. Récupération des informations générales
    
    st.title('Analyse Client et Octroi Crédit')

    # 2.1 Selection d'1 client

    id_client = st.sidebar.selectbox('Choisir ID Client :', df_desc.index)


    # 2.2 Affichage de ses informations générales dans la barre latérale

    st.sidebar.table(df_desc.loc[id_client][:9])



    # 3. Positionnement du client par rapport à clients défauts et clients solvables - données quant

    donnees = ['AGE', 'YEARS EMPLOYED', 'FAMILY MEMBERS','NUMBER CHILDREN','ANNUAL INCOME', 'AMOUNT ANNUITY',
               'AMOUNT CREDIT', 'GOODS VALUE']

    # 3.1 Chargement des données client

    data_client = df_desc.loc[id_client:id_client]
    

    # 3.2 Comparaison données clients et données moyennes

    st.header('1. Comparaison données client versus autres clients')
    df_pos = pd.concat([averages, data_client], join = 'inner').round(2)

    fig_1 = make_subplots(rows=4, cols=2,
                         subplot_titles=[donnees[i] for i in range(8)],
                          shared_xaxes = True,
                          vertical_spacing = 0.05
                          )

        # Ajout des graphes
    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[0]].iloc[i] for i in range(3)],
                              name=donnees[0]
                                ),
                   row=1, col=1, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Clients défault", "Notre Client"],
                             y=[df_pos[donnees[1]].iloc[i] for i in range(3)],
                              name=donnees[1]
                                ),
                   row=1, col=2, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[2]].iloc[i] for i in range(3)],
                              name=donnees[2]
                                ),
                   row=2, col=1, secondary_y=False)

    fig_1.add_trace( go.Scatter(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[3]].iloc[i] for i in range(3)],
                              name=donnees[3]
                                ),
                   row=2, col=2, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[4]].iloc[i] for i in range(3)],
                                marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[4]
                                ),
                   row=3, col=1, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[5]].iloc[i] for i in range(3)],
                            marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[5]
                            ),
                   row=3, col=2, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Clients défault", "Notre Client"],
                              y=[df_pos[donnees[6]].iloc[i] for i in range(3)],
                                marker= dict(color = ['mediumspringgreen', 'indianred','mediumblue']),
                              name=donnees[6]
                                ),
                   row=4, col=1, secondary_y=False)

    fig_1.add_trace( go.Bar(x=["Clients en règle", "Clients défault", "Notre Client"],
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

    data_client_select = data_clean.loc[id_client:id_client]
    client_pred = 100*model.predict_proba(data_client_select)[0][1]

    
    # 4.2 Même opération pour ses voisins

        # Récupération des indices des 10 plus proches voisins
    idx_voisins = tree.query(df_voisins.loc[id_client:id_client].fillna(0), k=10)[1][0]
        # Récupération des données 
    data_voisins = data_clean.iloc[idx_voisins]
        # Prédictions
    predict_voisins = 100*model.predict_proba(data_voisins).mean(axis=0)[1]
        # Moyennes des features des voisins
    mean_voisins = pd.DataFrame(data_voisins.mean(), columns=['voisins']).T
    

    # 4.3 Représentation graphique sous forme de jauge horizontale
    if st.sidebar.checkbox("Afficher Proba Défaut", False, key = 2):
        st.header('2. Probabilité de défaut pour le client {}'.format(id_client))
        fig0 = go.Figure(go.Indicator( mode = "number+gauge+delta",
                                       value = client_pred,
                                       domain = {'x': [0, 1], 'y': [0, 1]},
                                       delta = {'reference': predict_voisins,
                                                'increasing': {'color': 'red'},
                                                'decreasing' : {'color' : 'green'},
                                                'position': "top"},
                                       title = {'text':"<b>En %</b><br><span style='color: gray; font-size:0.8em'></span>", 'font': {"size": 14}},
                                       gauge = {'shape': "bullet",
                                                'axis': {'range': [None, 100]},
                                                'threshold': {
                                                    'line': {'color': "white", 'width': 3},
                                                    'thickness': 0.75, 'value': predict_voisins},
                                                'bgcolor': "white",
                                                'steps': [{'range': [0, 50], 'color': "lightgreen"},
                                                          {'range': [50, 60], 'color': "orange"},
                                                          {'range': [60, 100], 'color': "red"}],
                                                'bar': {'color': "darkblue"}}))
        fig0.update_layout(height = 250)

        st.plotly_chart(fig0)

        st.markdown('Proba défaut client sélectionné : **{0:.1f}%**'.format(client_pred))
        st.markdown('Proba défaut clients similaires : **{0:.1f}%** \
        (critères de similarité : âge, genre,situation familiale, éducation, profession)'.format(predict_voisins))



    # 5. Interprétation des résultats

    features = ['EXT_SOURCE_2' ,'EXT_SOURCE_3','TENOR','LTV','EXT_SOURCE_1','Life_employed','AMOUNT ANNUITY']
    
    feature_desc = {'EXT_SOURCE_2' : 'Score normalisé attribué par organisme indépendant #2',
                    'EXT_SOURCE_3' :  'Score normalisé attribué par organisme indépendant #3',
                     'TENOR': 'Durée du crédit demandé',
                     'LTV': 'Valeur du crédit par rapport au bien à financer',
                     'EXT_SOURCE_1' :  'Score normalisé attribué par organisme indépendant #1',
                     'Life_employed' : 'Années travaillées en pourcentage de l âge',
                    'AMOUNT ANNUITY' : 'Montant des annuités'}


    # 5.1 Aggrégation des données client, moyenne, et défaut / payeur dans un dataframe (une ligne par type de client)

    data_client_select = data_client_select.rename(columns={'AMT_ANNUITY': 'AMOUNT ANNUITY'})

    mean_voisins = mean_voisins.rename(columns={'AMT_ANNUITY': 'AMOUNT ANNUITY'})

    dfcomp = pd.concat([averages, mean_voisins, data_client_select], join = 'inner').round(2)

    if st.sidebar.checkbox("Afficher Explication Proba Def", False, key = 0):
        st.header('3. Variables principales influançant un octroi de crédit')
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




