import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Diabète",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🩺 Système de Prédiction de Diabète")
st.markdown("---")

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge le dataset de diabète"""
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour l'entraînement des modèles supervisés
@st.cache_data
def train_supervised_models(X_train, X_test, y_train, y_test):
    """Entraîne les modèles de classification"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=166, random_state=42, max_depth=8),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results

# Fonction pour le clustering
@st.cache_data
def perform_clustering(X_scaled, n_clusters=3):
    """Effectue le clustering avec différents algorithmes"""
    clustering_results = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    
    clustering_results['K-Means'] = {
        'labels': kmeans_labels,
        'silhouette_score': kmeans_silhouette,
        'model': kmeans
    }
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    if len(set(dbscan_labels)) > 1:
        dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    else:
        dbscan_silhouette = -1
    
    clustering_results['DBSCAN'] = {
        'labels': dbscan_labels,
        'silhouette_score': dbscan_silhouette,
        'model': dbscan
    }
    
    # CAH (Clustering Agglomératif Hiérarchique)
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_clustering.fit_predict(X_scaled)
    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    
    clustering_results['CAH'] = {
        'labels': agg_labels,
        'silhouette_score': agg_silhouette,
        'model': agg_clustering
    }
    
    return clustering_results

# Fonction pour traiter les valeurs aberrantes
@st.cache_data
def cap_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower,
                 np.where(df[column] > upper, upper, df[column]))
    

# Chargement des données
df = load_data()
if df is None:
    st.stop()

# Sidebar pour la navigation
st.sidebar.title("📋 Navigation")
pages = [
    "🏠 Accueil",
    "📊 Exploration des Données",
    "🔧 Prétraitement",
    "🤖 Apprentissage Supervisé",
    "🎯 Clustering (Non-supervisé)",
    "📈 Comparaison des Modèles",
    "🔮 Prédiction Interactive"
]

selected_page = st.sidebar.selectbox("Choisir une section", pages)

# Page d'accueil
if selected_page == "🏠 Accueil":
    st.markdown("""
    ## 🎯 Objectif du Projet
    Ce système permet de prédire la probabilité qu'un patient développe un diabète en se basant sur des données médicales.
    
    ## 📊 Dataset
    - **Source**: Dataset Pima Indians Diabetes
    - **Nombre d'échantillons**: {}
    - **Nombre de caractéristiques**: {}
    - **Variable cible**: Outcome (0 = non diabétique, 1 = diabétique)
    
    ## 🔧 Fonctionnalités
    - **Exploration des données**: Statistiques descriptives et visualisations
    - **Prétraitement**: Gestion des valeurs manquantes et normalisation
    - **Apprentissage supervisé**: 4 modèles de classification
    - **Apprentissage non supervisé**: Clustering avec K-Means, DBSCAN et CAH
    - **Prédiction interactive**: Interface pour prédire de nouveaux cas
    
    ## 📋 Description des Variables
    """.format(df.shape[0], df.shape[1] - 1))
    
    # Tableau des variables
    variables_info = {
        'Variable': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
        'Description': [
            'Nombre de grossesses',
            'Concentration en glucose',
            'Pression artérielle',
            'Épaisseur de la peau',
            'Niveau d\'insuline',
            'Indice de masse corporelle',
            'Risque génétique',
            'Âge du patient',
            'Cible (0 = non diabétique, 1 = diabétique)'
        ]
    }
    
    variables_df = pd.DataFrame(variables_info)
    st.table(variables_df)

# Page Exploration des Données
elif selected_page == "📊 Exploration des Données":
    st.header("📊 Exploration des Données")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre d'échantillons", df.shape[0])
    with col2:
        st.metric("Nombre de variables", df.shape[1] - 1)
    with col3:
        st.metric("Cas positifs", df['Outcome'].sum())
    with col4:
        st.metric("Cas négatifs", (df['Outcome'] == 0).sum())
    
    # Statistiques descriptives
    st.subheader("📈 Statistiques Descriptives")
    st.dataframe(df.describe())
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de la variable cible
        fig_pie = px.pie(
            values=df['Outcome'].value_counts().values,
            names=['Non Diabétique', 'Diabétique'],
            title="Répartition des Cas de Diabète",
            color_discrete_sequence=['lightblue', 'salmon']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Matrice de corrélation
        corr_matrix = df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distributions des variables
    st.subheader("📊 Distributions des Variables")
    selected_vars = st.multiselect(
        "Choisir les variables à visualiser",
        df.columns[:-1].tolist(),
        default=df.columns[:4].tolist()
    )
    
    if selected_vars:
        for var in selected_vars:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(
                    df, x=var, color='Outcome',
                    title=f"Distribution de {var}",
                    marginal="box",
                    color_discrete_sequence=['lightblue', 'salmon']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    df, x='Outcome', y=var,
                    title=f"Boxplot de {var} par Outcome",
                    color='Outcome',
                    color_discrete_sequence=['lightblue', 'salmon']
                )
                st.plotly_chart(fig_box, use_container_width=True)

# Page Prétraitement
elif selected_page == "🔧 Prétraitement":
    st.header("🔧 Prétraitement des Données")
    
    # Vérification des valeurs manquantes
    st.subheader("🔍 Analyse des Valeurs Manquantes")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ Aucune valeur manquante détectée!")
    else:
        st.warning("⚠️ Valeurs manquantes détectées:")
        st.dataframe(missing_values[missing_values > 0])
    
    # Détection des valeurs aberrantes
    st.subheader("🚨 Détection des Valeurs Aberrantes")
    
    # Certaines valeurs 0 peuvent être aberrantes pour certaines variables
    zero_counts = {}
    problematic_vars = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for var in problematic_vars:
        zero_count = (df[var] == 0).sum()
        zero_counts[var] = zero_count
    
    zero_df = pd.DataFrame(list(zero_counts.items()), columns=['Variable', 'Nombre de zéros'])
    st.dataframe(zero_df)
    
    if st.checkbox("Afficher les statistiques après traitement des zéros et Imputation des Valeurs Manquantes"):
        df_cleaned = df.copy()
        df_cleaned[problematic_vars] = df_cleaned[problematic_vars].replace(0, np.nan)

        for col in problematic_vars:
            if col in ["Insulin", "SkinThickness"]:
                value = df_cleaned[col].mean()
            else:
                value = df_cleaned[col].median()
        st.write("Statistiques après remplacement des zéros par la médiane:")
        st.dataframe(df_cleaned[problematic_vars].describe())
    
    # Normalisation
    st.subheader("📏 Normalisation des Données")
    if st.checkbox("Appliquer la normalisation StandardScaler"):
        X = df.drop('Outcome', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Données originales:**")
            st.dataframe(X.describe())
        with col2:
            st.write("**Données normalisées:**")
            st.dataframe(X_scaled_df.describe())

# Page Apprentissage Supervisé
elif selected_page == "🤖 Apprentissage Supervisé":
    st.header("🤖 Apprentissage Supervisé")
    
    # Préparation des données
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Options de prétraitement
    st.subheader("⚙️ Configuration")
    test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
    apply_scaling = st.checkbox("Appliquer la normalisation", value=True)
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if apply_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # Entraînement des modèles
    if st.button("🚀 Entraîner les Modèles", type="primary"):
        with st.spinner("Entraînement en cours..."):
            results = train_supervised_models(X_train_final, X_test_final, y_train, y_test)
            st.session_state.supervised_results = results
            st.session_state.X_test = X_test_final
            st.session_state.y_test = y_test
    
    # Affichage des résultats
    if 'supervised_results' in st.session_state:
        results = st.session_state.supervised_results
        
        st.subheader("📊 Résultats des Modèles")
        
        # Métriques de performance
        metrics_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'Précision': [results[model]['accuracy'] for model in results.keys()]
        }).sort_values('Précision', ascending=False)
        
        fig_metrics = px.bar(
            metrics_df, x='Modèle', y='Précision',
            title="Comparaison des Précisions",
            color='Précision',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Matrices de confusion
        st.subheader("🎯 Matrices de Confusion")
        selected_model = st.selectbox("Choisir un modèle", list(results.keys()))
        
        if selected_model:
            conf_matrix = results[selected_model]['confusion_matrix']
            
            fig_cm = px.imshow(
                conf_matrix,
                text_auto=True,
                aspect="auto",
                title=f"Matrice de Confusion - {selected_model}",
                labels=dict(x="Prédiction", y="Réalité"),
                x=['Non Diabétique', 'Diabétique'],
                y=['Non Diabétique', 'Diabétique']
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Rapport de classification
            st.subheader("📋 Rapport de Classification")
            class_report = results[selected_model]['classification_report']
            report_df = pd.DataFrame(class_report).iloc[:-1, :].T
            st.dataframe(report_df)
        
        # Courbes ROC
        st.subheader("📈 Courbes ROC")
        fig_roc = go.Figure()
        
        for model_name in results.keys():
            if results[model_name]['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(st.session_state.y_test, results[model_name]['probabilities'])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {roc_auc:.2f})'
                ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Ligne de base'
        ))
        
        fig_roc.update_layout(
            title='Courbes ROC',
            xaxis_title='Taux de Faux Positifs',
            yaxis_title='Taux de Vrais Positifs'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

# Page Clustering
elif selected_page == "🎯 Clustering (Non-supervisé)":
    st.header("🎯 Apprentissage Non Supervisé - Clustering")
    
    # Préparation des données
    X = df.drop('Outcome', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configuration du clustering
    st.subheader("⚙️ Configuration")
    n_clusters = st.slider("Nombre de clusters (K-Means, CAH)", 2, 8, 3)
    
    # Application du clustering
    if st.button("🎯 Appliquer le Clustering", type="primary"):
        with st.spinner("Clustering en cours..."):
            clustering_results = perform_clustering(X_scaled, n_clusters)
            st.session_state.clustering_results = clustering_results
            st.session_state.X_scaled = X_scaled
    
    # Affichage des résultats
    if 'clustering_results' in st.session_state:
        clustering_results = st.session_state.clustering_results
        X_scaled = st.session_state.X_scaled
        
        # Scores de silhouette
        st.subheader("📊 Scores de Silhouette")
        silhouette_df = pd.DataFrame({
            'Algorithme': list(clustering_results.keys()),
            'Score de Silhouette': [clustering_results[alg]['silhouette_score'] for alg in clustering_results.keys()]
        }).sort_values('Score de Silhouette', ascending=False)
        
        fig_silhouette = px.bar(
            silhouette_df, x='Algorithme', y='Score de Silhouette',
            title="Comparaison des Scores de Silhouette",
            color='Score de Silhouette',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Visualisation PCA
        st.subheader("🎨 Visualisation des Clusters (PCA)")
        
        # Application de la PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        selected_algorithm = st.selectbox("Choisir l'algorithme de clustering", list(clustering_results.keys()))
        
        if selected_algorithm:
            labels = clustering_results[selected_algorithm]['labels']
            
            # Création du DataFrame pour Plotly
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': labels,
                'Outcome': df['Outcome'].values
            })
            
            col1, col2 = st.columns(2)

            color_scales = {
                'DBSCAN': 'viridis',
                'K-Means': 'plasma',
                'CAH': 'cividis'
                }

            with col1:
                fig_cluster = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Cluster',
                    title=f'Clusters - {selected_algorithm}',
                    color_continuous_scale=color_scales.get(selected_algorithm, 'viridis')
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with col2:
                fig_outcome = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Outcome',
                    title='Données Réelles (Outcome)',
                    color_discrete_sequence=['lightblue', 'salmon']
                )
                st.plotly_chart(fig_outcome, use_container_width=True)
        
        # Méthode du coude pour K-Means
        st.subheader("📐 Méthode du Coude (K-Means)")
        if st.checkbox("Afficher l'analyse du coude"):
            with st.spinner("Calcul en cours..."):
                inertias = []
                k_range = range(1, 11)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                
                fig_elbow = px.line(
                    x=list(k_range), y=inertias,
                    title="Méthode du Coude",
                    markers=True
                )
                fig_elbow.update_layout(
                    xaxis_title="Nombre de Clusters",
                    yaxis_title="Inertie"
                )
                st.plotly_chart(fig_elbow, use_container_width=True)

# Page Comparaison des Modèles
elif selected_page == "📈 Comparaison des Modèles":
    st.header("📈 Comparaison des Modèles")
    
    if 'supervised_results' in st.session_state and 'clustering_results' in st.session_state:
        supervised_results = st.session_state.supervised_results
        clustering_results = st.session_state.clustering_results
        
        # Comparaison des modèles supervisés
        st.subheader("🤖 Modèles Supervisés")
        
        # Tableau de comparaison
        comparison_data = []
        for model_name, results in supervised_results.items():
            accuracy = results['accuracy']
            precision = results['classification_report']['1']['precision']
            recall = results['classification_report']['1']['recall']
            f1 = results['classification_report']['1']['f1-score']
            
            comparison_data.append({
                'Modèle': model_name,
                'Précision': f"{accuracy:.3f}",
                'Precision': f"{precision:.3f}",
                'Rappel': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparaison des métriques
        metrics = ['Précision', 'Precision', 'Rappel', 'F1-Score']
        selected_metric = st.selectbox("Choisir une métrique à comparer", metrics)
        
        if selected_metric == 'Précision':
            values = [supervised_results[model]['accuracy'] for model in supervised_results.keys()]
        elif selected_metric == 'Precision':
            values = [supervised_results[model]['classification_report']['1']['precision'] for model in supervised_results.keys()]
        elif selected_metric == 'Rappel':
            values = [supervised_results[model]['classification_report']['1']['recall'] for model in supervised_results.keys()]
        else:  # F1-Score
            values = [supervised_results[model]['classification_report']['1']['f1-score'] for model in supervised_results.keys()]
        
        fig_comparison = px.bar(
            x=list(supervised_results.keys()),
            y=values,
            title=f"Comparaison - {selected_metric}",
            color=values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Comparaison clustering
        st.subheader("🎯 Modèles de Clustering")
        
        clustering_comparison = pd.DataFrame({
            'Algorithme': list(clustering_results.keys()),
            'Score de Silhouette': [clustering_results[alg]['silhouette_score'] for alg in clustering_results.keys()],
            'Nombre de Clusters': [
                len(set(clustering_results[alg]['labels'])) for alg in clustering_results.keys()
            ]
        })
        st.dataframe(clustering_comparison, use_container_width=True)
        
        # Recommandations
        st.subheader("💡 Recommandations")
        
        # Meilleur modèle supervisé
        best_supervised = max(supervised_results.keys(), key=lambda x: supervised_results[x]['accuracy'])
        best_accuracy = supervised_results[best_supervised]['accuracy']
        
        # Meilleur clustering
        best_clustering = max(clustering_results.keys(), key=lambda x: clustering_results[x]['silhouette_score'])
        best_silhouette = clustering_results[best_clustering]['silhouette_score']
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Meilleur modèle supervisé**: {best_supervised}")
            st.info(f"Précision: {best_accuracy:.3f}")
        
        with col2:
            st.success(f"🏆 **Meilleur clustering**: {best_clustering}")
            st.info(f"Score de Silhouette: {best_silhouette:.3f}")
    
    else:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles supervisés et effectuer le clustering.")

# Page Prédiction Interactive
elif selected_page == "🔮 Prédiction Interactive":
    st.header("🔮 Prédiction Interactive")
    
    if 'supervised_results' in st.session_state:
        supervised_results = st.session_state.supervised_results
        
        st.subheader("🎛️ Saisie des Données Patient")
        
        # Interface de saisie
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Concentration en glucose", min_value=0, max_value=250, value=120)
            blood_pressure = st.number_input("Pression artérielle", min_value=0, max_value=200, value=80)
            skin_thickness = st.number_input("Épaisseur de la peau", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Niveau d'insuline", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Risque génétique", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Âge", min_value=1, max_value=120, value=30)
        
        # Bouton de prédiction
        if st.button("🔮 Prédire", type="primary"):
            # Préparation des données
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                  insulin, bmi, diabetes_pedigree, age]])
            
            # Normalisation si nécessaire (assumant que les modèles ont été entraînés avec normalisation)
            if 'apply_scaling' not in st.session_state:
                st.session_state.apply_scaling = True
            
            if st.session_state.apply_scaling:
                X_original = df.drop('Outcome', axis=1)
                scaler = StandardScaler()
                scaler.fit(X_original)
                input_data_scaled = scaler.transform(input_data)
                input_final = input_data_scaled
            else:
                input_final = input_data
            
            # Prédictions
            st.subheader("📊 Résultats des Prédictions")
            
            predictions = {}
            probabilities = {}
            
            for model_name, model_info in supervised_results.items():
                model = model_info['model']
                pred = model.predict(input_final)[0]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_final)[0][1]
                    probabilities[model_name] = proba
                predictions[model_name] = pred
            
            # Affichage des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prédictions")
                for model_name, pred in predictions.items():
                    if pred == 1:
                        st.error(f"❌ {model_name}: **Diabétique**")
                    else:
                        st.success(f"✅ {model_name}: **Non Diabétique**")
            
            with col2:
                if probabilities:
                    st.subheader("📈 Probabilités")
                    prob_df = pd.DataFrame({
                        'Modèle': list(probabilities.keys()),
                        'Probabilité de Diabète': list(probabilities.values())
                    })
                    
                    fig_prob = px.bar(
                        prob_df, x='Modèle', y='Probabilité de Diabète',
                        title="Probabilités de Diabète par Modèle",
                        color='Probabilité de Diabète',
                        color_continuous_scale='Reds',
                        range_y=[0, 1]
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
            
            # Consensus des modèles
            st.subheader("🏛️ Consensus des Modèles")
            positive_predictions = sum(predictions.values())
            total_models = len(predictions)
            consensus_percentage = (positive_predictions / total_models) * 100
            
            if consensus_percentage >= 75:
                st.error(f"⚠️ **Risque Élevé**: {positive_predictions}/{total_models} modèles prédisent un diabète ({consensus_percentage:.1f}%)")
            elif consensus_percentage >= 50:
                st.warning(f"⚡ **Risque Modéré**: {positive_predictions}/{total_models} modèles prédisent un diabète ({consensus_percentage:.1f}%)")
            else:
                st.success(f"✅ **Risque Faible**: {positive_predictions}/{total_models} modèles prédisent un diabète ({consensus_percentage:.1f}%)")
            
            # Recommandations médicales
            st.subheader("🩺 Recommandations")
            
            recommendations = []
            
            if glucose > 140:
                recommendations.append("🔴 Niveau de glucose élevé - Consulter un médecin")
            elif glucose > 100:
                recommendations.append("🟡 Glucose légèrement élevé - Surveiller régulièrement")
            
            if bmi > 30:
                recommendations.append("🔴 IMC indique une obésité - Consulter un nutritionniste")
            elif bmi > 25:
                recommendations.append("🟡 IMC en surpoids - Considérer un régime équilibré")
            
            if blood_pressure > 140:
                recommendations.append("🔴 Pression artérielle élevée - Surveillance médicale recommandée")
            elif blood_pressure > 120:
                recommendations.append("🟡 Pression artérielle légèrement élevée - Surveiller")
            
            if age > 45:
                recommendations.append("🟡 Âge de risque - Dépistage régulier recommandé")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ Paramètres dans les normes générales")
            
            # Graphique radar des paramètres
            st.subheader("🎯 Profil du Patient")
            
            # Normaliser les valeurs pour le graphique radar
            patient_values = [pregnancies, glucose/200, blood_pressure/140, skin_thickness/50,
                            insulin/300, bmi/40, diabetes_pedigree, age/80]
            
            categories = ['Grossesses', 'Glucose', 'Pression', 'Peau',
                         'Insuline', 'IMC', 'Génétique', 'Âge']
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_values,
                theta=categories,
                fill='toself',
                name='Patient',
                line_color='red'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Profil Radar du Patient"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    else:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles dans la section 'Apprentissage Supervisé'.")

# Section Informations et Aide
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Informations")
st.sidebar.info("""
**Modèles Utilisés:**
- Random Forest
- Régression Logistique  
- SVM
- K-Nearest Neighbors

**Clustering:**
- K-Means
- DBSCAN
- CAH (Clustering Hiérarchique)

**Métriques:**
- Précision
- Matrice de confusion
- Courbe ROC
- Score de Silhouette
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Instructions")
st.sidebar.markdown("""
1. **Exploration**: Analysez les données
2. **Prétraitement**: Vérifiez la qualité
3. **Supervisé**: Entraînez les modèles
4. **Clustering**: Analysez les groupes
5. **Comparaison**: Évaluez les performances
6. **Prédiction**: Testez de nouveaux cas
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🩺 Système de Prédiction de Diabète | Développé avec Streamlit</p>
</div>
""", unsafe_allow_html=True)