import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc

import folium
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import streamlit_folium as sf
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import Image
import plotly.figure_factory as ff
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# Fonction pour le chargement du dataset
@st.cache_data
def load_data():
    data = pd.read_excel('./data/telecom_churn_cleaned.xlsx')
    return data

data = load_data()

# Fonction pour préparer les données
def prepare_data(data):
    # Calcul de la proportion de répondants pour chaque État
    count_by_state = data["State"].value_counts()
    proportion_by_state = count_by_state / count_by_state.sum() * 100
    data["proportion_repondants"] = data["State"].map(proportion_by_state)  
  
    # Calculer la proportion de churn pour chaque État
    churn_by_state = data.groupby('State')['Churn'].value_counts().unstack()
    churn_by_state['Total'] = churn_by_state['False.'] + churn_by_state['True.']
    churn_by_state['Churn_Proportion_True'] = (churn_by_state['True.'] / churn_by_state['Total']) * 100
    churn_by_state['Churn_Proportion_False'] = (churn_by_state['False.'] / churn_by_state['Total']) * 100

    # Ajouter la proportion de churn par État au dataframe principal
    data['proportion_churn_true'] = data['State'].map(churn_by_state['Churn_Proportion_True'])
    data['proportion_churn_false'] = data['State'].map(churn_by_state['Churn_Proportion_False'])

    # Convertir les colonnes d'objets en catégories
    cols_to_convert = ['State', 'Intl_Plan', 'VMail_Plan', 'Churn']
    for col in cols_to_convert:
        data[col] = data[col].astype('category')

    return data

data = prepare_data(data)

def create_client_data(data, columns, account_length, intl_plan, vmail_plan, vmail_message, 
                       day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, 
                       night_mins, night_calls, night_charge, intl_mins, intl_calls, 
                       intl_charge, custserv_calls):
    client_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    client_data['Account_Length'] = account_length
    client_data['Intl_Plan_Yes'] = 1 if intl_plan == 'Yes' else 0
    client_data['VMail_Plan_Yes'] = 1 if vmail_plan == 'Yes' else 0
    client_data['VMail_Message'] = vmail_message
    client_data['Day_Mins'] = day_mins
    client_data['Day_Calls'] = day_calls
    client_data['Day_Charge'] = day_charge
    client_data['Eve_Mins'] = eve_mins
    client_data['Eve_Calls'] = eve_calls
    client_data['Eve_Charge'] = eve_charge
    client_data['Night_Mins'] = night_mins
    client_data['Night_Calls'] = night_calls
    client_data['Night_Charge'] = night_charge
    client_data['Intl_Mins'] = intl_mins
    client_data['Intl_Calls'] = intl_calls
    client_data['Intl_Charge'] = intl_charge
    client_data['CustServ_Calls'] = custserv_calls

# #     return client_data
# def create_client_data(data, columns, account_length, intl_plan, vmail_plan, vmail_message, day_mins, day_calls, eve_mins, eve_calls,
#                        night_mins, night_calls, intl_mins, intl_calls, intl_charge, custserv_calls, custserv_charge):
#     # Create a new dataframe for the input data
#     client_data = pd.DataFrame(columns=columns)

#     # Create a dictionary with the input data
#     client_dict = {'Account Length': account_length,
#                    'International Plan': intl_plan,
#                    'Voice Mail Plan': vmail_plan,
#                    'Number Vmail Messages': vmail_message,
#                    'Total Day Minutes': day_mins,
#                    'Total Day Calls': day_calls,
#                    'Total Eve Minutes': eve_mins,
#                    'Total Eve Calls': eve_calls,
#                    'Total Night Minutes': night_mins,
#                    'Total Night Calls': night_calls,
#                    'Total Intl Minutes': intl_mins,
#                    'Total Intl Calls': intl_calls,
#                    'Total Intl Charge': intl_charge,
#                    'Customer Service Calls': custserv_calls,
#                    'Total Customer Service Calls': custserv_charge}

#     # Append the input data to the new dataframe
#     client_data = client_data.append(client_dict, ignore_index=True)

    # Encode categorical variables
    client_data = pd.get_dummies(client_data, drop_first=True)

    # Reorder the columns
    client_data = client_data.reindex(columns=columns, fill_value=0)

    return client_data


# Fonction pour entraîner et évaluer un modèle de prédiction
def train_and_evaluate_model(data, model_name):
    # Préparation des données pour l'entraînement et l'évaluation
    X = data.drop(['Churn', 'Phone', 'State'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())
    y = data['Churn'].cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    # Entraînement du modèle
    if model_name == 'Random Forest':
        clf = RandomForestClassifier(random_state=42)

    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=42)

    elif model_name == 'SVM':
        clf = SVC(random_state=42)
    elif model_name == 'XGBoost':
        clf = xgb.XGBClassifier(random_state=42)
       
    clf.fit(X_train, y_train)
    # Évaluation du modèle
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, report, X_test, y_test, X_train, y_train, X

def plot_interactive_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"Courbe ROC (AUC = {roc_auc:.2f})"
    )

    trace1 = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Aléatoire',
        line=dict(dash='dash')
    )

    layout = go.Layout(
        title=f"Courbe ROC pour le modèle {model_name}",
        xaxis=dict(title='Taux de faux positifs'),
        yaxis=dict(title='Taux de vrais positifs'),
        showlegend=True
    )

    fig = go.Figure(data=[trace0, trace1], layout=layout)
    st.plotly_chart(fig)


# Interface utilisateur Streamlit
st.title("Analyse du désabonnement des clients des télécommunications \n Par Eddy Rigaud PharmD candidate / Marketing & Data Analytics at Neoma BS")



page = st.sidebar.radio("Choisissez une page", ['Exploration des données (EDA)', 'Prédiction'])


if page == 'Exploration des données (EDA)':
    
    
    st.subheader("Aperçu des données")
    st.write(data.head())

    st.subheader("Statistiques descriptives")
    summary = (data[[i for i in data.columns]].
           describe().transpose().reset_index())

    summary = summary.rename(columns = {"index" : "feature"})
    summary = np.around(summary,3)      
    st.write(summary)

    # Distribution de l'attrition (Churn)
    st.subheader("Proportion de Churn")
    fig = px.pie(data, names='Churn', title='Distribution du Churn')
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


    st.subheader("Charges par minutes pour chaques Catégories d'appels")
    day_charge_per_minutes = data['Day_Charge'].sum()/data['Day_Mins'].sum()
    eve_charge_per_minutes = data['Eve_Charge'].sum()/data['Eve_Mins'].sum()
    night_charge_per_minutes = data['Night_Charge'].sum()/data['Night_Mins'].sum()
    int_charge_per_minutes = data['Intl_Charge'].sum()/data['Intl_Mins'].sum()
    fig = go.Figure()

    fig.add_trace(go.Bar(x=['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'],
                     y=[day_charge_per_minutes, eve_charge_per_minutes, night_charge_per_minutes, int_charge_per_minutes],
                     text=['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'],
                     textposition='auto',
                     marker=dict(color=['orange', 'blue', 'green', 'red'])))

    fig.update_layout(
        title="Charges par minutes pour chaques Catégories d'appels",
        xaxis_title="Catégories d'appels",
        yaxis_title="Charges par minute",
    )

    # Affichez le diagramme dans Streamlit
    st.plotly_chart(fig)

    #Ajout d'un metrics sur la page pour voir les charges moyennes par users
    arpu = data[['Day_Charge', 'Eve_Charge', 'Night_Charge', 'Intl_Charge']].sum(axis=1).mean()
    st.metric(label="Charges moyenne par utilisateur", value=f"{arpu:.2f} $", delta=None)

    st.subheader("Histogrammes de Churn en Fonction des Etats, Nombre d'appels au service au client, Option Messagerie Vocale, Option appels à l'étranger")
    for col in ['State', 'CustServ_Calls', "VMail_Plan", "Intl_Plan"]:
        fig = px.histogram(data, x=col, color='Churn', nbins=len(data[col].unique()), histnorm='percent')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        st.plotly_chart(fig)

    # Matrix de Corrélation entre les variables 
    # On va mettre en évidence ce que l'on a étudié sur le notebook cad La corrélation entre les variable de Minutes et de Charges dans notre dataset. 
    # Cela nous sera utile pour les prédiction lors de l'utilisation du model.
    st.subheader("Corrélations entre les variables")
    correlations = data.corr()
    fig = go.Figure(go.Heatmap(z=correlations, x=correlations.columns, y=correlations.columns, colorscale='viridis'))
    st.plotly_chart(fig)

    # Carte Interactive Users // Churn

    st.subheader("Répartition géographique")
    option = st.selectbox("Choisir la répartition", ["Utilisateurs", "Churn"])

    if option == "Utilisateurs":
        st.write("Voici la répartition géographique des users par État: \n Vous pouvez cliquer sur les boutons pour avoir plus de précision.")

        # Code pour créer la carte avec la répartition des utilisateurs
        # Utilisez le code que vous avez fourni pour créer la carte avec les marqueurs pour les utilisateurs
        m = folium.Map(location=[39.833333, -98.583333], zoom_start=4, tiles="cartodb dark_matter")

        # Ajoutez les marqueurs à la carte
        marker_cluster = MarkerCluster().add_to(m)

        for i in range(len(data)):
            lat = data.iloc[i]['Latitude']
            long = data.iloc[i]['Longitude']
            radius = 5
            popup_text = """Country : {}<br>% of Users : {:.2f}%<br>"""
            popup_text = popup_text.format(data.iloc[i]['state'], data.iloc[i]['proportion_repondants'])
            folium.CircleMarker(location=[lat, long], radius=radius, popup=popup_text, fill=True).add_to(marker_cluster)

        # Affiche la carte dans Streamlit
        sf.folium_static(m)

    elif option == "Churn":
    # Code pour créer la carte avec la répartition du churn
    # Modifiez le code que vous avez fourni pour créer la carte avec les marqueurs pour le churn
    # Assurez-vous d'avoir une colonne dans votre dataframe avec la proportion de churn par État
        st.write("Voici la répartition géographique du churn par État: \n Vous pouvez selectionner la condition Chun ou Non Churn dans les options de layers en haut a droite de la map")
        m = folium.Map(location=[39.833333, -98.583333], zoom_start=4, tiles="cartodb dark_matter")

        # Créez les groupes de marqueurs pour les churn et les non-churn
        churn_group = folium.FeatureGroup(name='Churn')
        no_churn_group = folium.FeatureGroup(name='Non Churn')

        # Ajoutez les marqueurs à la carte
        marker_cluster = MarkerCluster().add_to(m)

        for i in range(len(data)):
            lat = data.iloc[i]['Latitude']
            long = data.iloc[i]['Longitude']
            churn_popup_text = f"State : {data.iloc[i]['state']}<br>Churn Proportion : {data.iloc[i]['proportion_churn_true']:.2f}%<br>"
            no_churn_popup_text = f"State : {data.iloc[i]['state']}<br>Non Churn Proportion : {data.iloc[i]['proportion_churn_false']:.2f}%<br>"
            churn_marker = folium.CircleMarker(location=[lat, long], radius=5, popup=churn_popup_text, color='red', fill=True)
            no_churn_marker = folium.CircleMarker(location=[lat, long], radius=5, popup=no_churn_popup_text, color='green', fill=True)

            # Ajoutez les marqueurs rouges et verts avec les proportions respectives pour chaque État
            churn_group.add_child(churn_marker)
            no_churn_group.add_child(no_churn_marker)

        # Ajoutez les groupes de marqueurs à la carte et ajoutez un contrôle de calques pour afficher/masquer les groupes
        m.add_child(churn_group)
        m.add_child(no_churn_group)
        folium.LayerControl().add_to(m)
        
        # Affiche la carte dans Streamlit
        sf.folium_static(m)
    # elif option == "Churn":
    #     st.write("Voici la répartition géographique du churn par État:")
    #     m = folium.Map(location=[39.833333, -98.583333], zoom_start=4, tiles="cartodb dark_matter")

    #     for i in range(len(data)):
    #         lat = data.iloc[i]['Latitude']
    #         long = data.iloc[i]['Longitude']
    #         churn_proportion_true = data.iloc[i]['proportion_churn_true']
    #         churn_proportion_false = data.iloc[i]['proportion_churn_false']

    #         # Choisir la couleur en fonction de la proportion de churn
    #         if 27 >= churn_proportion_true:
    #             color = 'red'
    #             popup_text = f"State : {data.iloc[i]['state']}<br>Churn Proportion : {churn_proportion_true:.2f}%<br>"
    #         else:
    #             color = 'green'
    #             popup_text = f"State : {data.iloc[i]['state']}<br>Non Churn Proportion : {churn_proportion_false:.2f}%<br>"

    #         # Ajouter le marqueur avec la couleur et l'opacité appropriées
    #         marker = folium.CircleMarker(
    #             location=[lat, long],
    #             radius=5,
    #             popup=popup_text,
    #             color=color,
    #             fill=True,
    #             fill_opacity=0.2
    #         )
    #         marker.add_to(m)

    #     # Affiche la carte dans Streamlit
    #     sf.folium_static(m)


# PREDICTION 
elif page == 'Prédiction':

    st.header("Prédiction de la désabonnement des clients")

    st.subheader("Sélectionnez le modèle de prédiction")
    model_name = st.selectbox("Modèle", ['Random Forest', 'Decision Tree', 'SVM', 'XGBoost'])

    clf, report, X_test, y_test, X_train, y_train, X = train_and_evaluate_model(data, model_name)

    st.subheader("Rapport de classification")
    st.write(pd.DataFrame(report).transpose())

    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, clf.predict(X_test))
    fig_cm = ff.create_annotated_heatmap(z=cm, x=['Non Churn', 'Churn'], y=['Non Churn', 'Churn'], colorscale='viridis')
    st.plotly_chart(fig_cm)

    st.subheader("Prédire la désabonnement pour un client spécifique")
    st.write("Veuillez entrer les informations du client :")

    # Saisie des informations du client (supprimez les variables Charges)
    # account_length = st.number_input("Durée de vie du compte", min_value=1, max_value=500, value=100)
    intl_plan = st.selectbox("Plan International", ['No', 'Yes'])
    vmail_plan = st.selectbox("Plan de Messagerie Vocale", ['No', 'Yes'])
    # vmail_message = st.number_input("Nombre d'emails reçus", min_value=0, max_value=60, value=20)
    # day_mins = st.number_input("Temps d'appel par jour en minutes", min_value=0.0, max_value=500.0, value=150.0)
    # day_calls = st.number_input("Nombre d'appels par jour", min_value=0, max_value=200, value=100)
    # eve_mins = st.number_input("Temps d'appel par le soir en minutes", min_value=0.0, max_value=500.0, value=150.0)
    # eve_calls = st.number_input("Nombre d'appels le soir", min_value=0, max_value=200, value=100)
    # night_mins = st.number_input("Temps d'appel par la nuit en minutes", min_value=0.0, max_value=500.0, value=150.0)
    # night_calls = st.number_input("Nombre d'appels la nuit", min_value=0, max_value=200, value=100)
    # intl_mins = st.number_input("Temps d'appel à l'international en minutes", min_value=0.0, max_value=60.0, value=10.0)
    # intl_calls = st.number_input("Nombre d'appels à l'international", min_value=0, max_value=20, value=5)
    # custserv_calls = st.number_input("Nombre d'appels au service client", min_value=0, max_value=10, value=2)
    # night_charge = st.number_input("Coût des appels la nuit", min_value=0.0, max_value=100.0, value=10.0)
    # intl_charge = st.number_input("Coût des appels à l'international", min_value=0.0, max_value=10.0, value=2.0)
    # eve_charge = st.number_input("Coût des appels le soir", min_value=0.0, max_value=100.0, value=20.0)
    # day_charge = st.number_input("Coût des appels par jour", min_value=0.0, max_value=100.0, value=30.0)
    account_length = st.slider("Durée de vie du compte", min_value=int(X['Account_Length'].min()), max_value=int(X['Account_Length'].max()), value=100)
    vmail_message = st.slider("Nombre d'emails reçus", min_value=int(X['VMail_Message'].min()), max_value=int(X['VMail_Message'].max()), value=20)
    day_mins = st.slider("Temps d'appel par jour en minutes", min_value=float(X['Day_Mins'].min()), max_value=float(X['Day_Mins'].max()), value=150.0)
    day_calls = st.slider("Nombre d'appels par jour", min_value=int(X['Day_Calls'].min()), max_value=int(X['Day_Calls'].max()), value=100)
    eve_mins = st.slider("Temps d'appel par le soir en minutes", min_value=float(X['Eve_Mins'].min()), max_value=float(X['Eve_Mins'].max()), value=150.0)
    eve_calls = st.slider("Nombre d'appels le soir", min_value=int(X['Eve_Calls'].min()), max_value=int(X['Eve_Calls'].max()), value=100)
    night_mins = st.slider("Temps d'appel par la nuit en minutes", min_value=float(X['Night_Mins'].min()), max_value=float(X['Night_Mins'].max()), value=150.0)
    night_calls = st.slider("Nombre d'appels la nuit", min_value=int(X['Night_Calls'].min()), max_value=int(X['Night_Calls'].max()), value=100)
    intl_mins = st.slider("Temps d'appel à l'international en minutes", min_value=float(X['Intl_Mins'].min()), max_value=float(X['Intl_Mins'].max()), value=10.0)
    intl_calls = st.slider("Nombre d'appels à l'international", min_value=int(X['Intl_Calls'].min()), max_value=int(X['Intl_Calls'].max()), value=5)
    custserv_calls = st.slider("Nombre d'appels au service client", min_value=int(X['CustServ_Calls'].min()), max_value=int(X['CustServ_Calls'].max()), value=2)
    night_charge = st.slider("Coût des appels la nuit", min_value=float(X['Night_Charge'].min()), max_value=float(X['Night_Charge'].max()), value=10.0)
    intl_charge = st.slider("Coût des appels à l'international", min_value=float(X['Intl_Charge'].min()), max_value=float(X['Intl_Charge'].max()), value=2.0)
    eve_charge = st.slider("Coût des appels le soir", min_value=float(X['Eve_Charge'].min()), max_value=float(X['Eve_Charge'].max()), value=20.0)
    day_charge = st.slider("Coût des appels par jour", min_value=float(X['Day_Charge'].min()), max_value=float(X['Day_Charge'].max()), value=30.0)

    if st.button("Prédire la désabonnement"):
        client_data = create_client_data(data, X_test.columns, account_length, intl_plan, vmail_plan, vmail_message, 
                                     day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, 
                                     night_calls, night_charge, intl_mins, intl_calls, intl_charge, custserv_calls)
        
    # Inutile d'utiliser pd.get_dummies sur client_data, car les colonnes sont déjà correctement encodées
        prediction = clf.predict(client_data)
        st.subheader("Courbe ROC")
        plot_interactive_roc_curve(clf, X_test, y_test)
        
            
        if prediction[0] == 1:
            st.error("Ce client est susceptible de se désabonner.")
        else:
            st.success("Ce client n'est pas susceptible de se désabonner.")
        
        

    # if st.checkbox("Afficher l'arbre de décision interactif"):
    #     if model_name == 'Random Forest':
    #         st.subheader("Arbre de décision interactif pour Random Forest")
    #         tree = clf.estimators_[0]
    #     elif model_name == 'Decision Tree':
    #         st.subheader("Arbre de décision interactif pour Decision Tree")
    #         tree = clf
    #     elif model_name == 'XGBoost':
    #         st.subheader("Arbre de décision interactif pour XGBoost")
    #         tree_num = 0

    #     if model_name == 'Random Forest' or model_name == 'Decision Tree':
    #         viz = dtreeviz(tree, X_train, y_train,
    #                     target_name='Churn',
    #                     feature_names=X_test.columns,
    #                     class_names=['Non Churn', 'Churn'])
    #         st.write(ff.create_tree(viz))
    #     elif model_name == 'XGBoost':
    #         st.warning("L'affichage d'un arbre de décision interactif pour XGBoost n'est pas pris en charge.")





   
