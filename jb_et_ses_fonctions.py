import pandas as pd

 # Colonnes de satisfaction (scores attendus entre 0 et 5)
satisfaction_columns = [
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness'
    ]

def diagnostic_valeurs_aberrantes(df):
    """
    Affiche un diagnostic des valeurs aberrantes dans les données :
    - Valeurs hors de l’échelle [0, 5] dans les colonnes de satisfaction
    - Distances de vol > 20 000 km
    - Retards (départ/arrivée) > 1 440 minutes
    """

    
    # 1. Compter les valeurs hors [0, 5]
    out_of_range_counts = {}
    for col in satisfaction_columns:
        count = df[~df[col].between(0, 5)].shape[0]
        out_of_range_counts[col] = count

    outliers_df = pd.DataFrame.from_dict(out_of_range_counts, orient='index', columns=['Valeurs hors [0-5]'])

    print("\n🔍 Valeurs hors échelle [0-5] dans les variables de satisfaction :")
    print(outliers_df.sort_values(by='Valeurs hors [0-5]', ascending=False))


    # 2. Distance de vol > 20 000 km
    flight_distance_outliers = df[df['Flight Distance'] > 20000]
    print(f"\n✈️ Nombre de vols avec une distance > 20 000 km : {flight_distance_outliers.shape[0]}")


    # 3. Retards > 1440 minutes
    departure_delay_outliers = df[df['Departure Delay in Minutes'] > 1440]
    arrival_delay_outliers = df[df['Arrival Delay in Minutes'] > 1440]

    print(f"\n⏱️ Nombre de retards au départ > 1 440 minutes : {departure_delay_outliers.shape[0]}")
    print(f"⏱️ Nombre de retards à l’arrivée > 1 440 minutes : {arrival_delay_outliers.shape[0]}")




def nettoyer_valeurs_aberrantes(df):
    """
    Nettoie les données en supprimant :
    - Les lignes avec des valeurs hors [0, 5] dans les colonnes de satisfaction
    - Les lignes avec 'Flight Distance' > 20 000 km
    - Les lignes avec 'Departure Delay in Minutes' ou 'Arrival Delay in Minutes' > 1 440 minutes

    Retourne un DataFrame nettoyé.
    """

    # Copie de sécurité pour ne pas modifier l'original
    df_clean = df.copy()

    # Supprimer les lignes avec des scores hors de [0, 5]
    for col in satisfaction_columns:
        df_clean = df_clean[df_clean[col].between(0, 5)]

    # Supprimer les lignes avec des distances > 20 000 km
    df_clean = df_clean[df_clean['Flight Distance'] <= 20000]

    # Supprimer les retards > 1 440 minutes
    df_clean = df_clean[df_clean['Departure Delay in Minutes'] <= 1440]
    df_clean = df_clean[df_clean['Arrival Delay in Minutes'] <= 1440]

    # Retourner le DataFrame nettoyé
    return df_clean



import numpy as np

def Imput_Valeur_Manquate ( Data, columns_Num, columns_Cat, Imput_Num= "median", 
                          Imput_Cat= "Mode" ) :
    for col in columns_Num :
        if Imput_Num == "median" :
            Change = np.median(Data[col].dropna()) 
        else:  
            Change = int(Imput_Num)
        Data[col] = Data[col].fillna(Change)
    for col in columns_Cat :
        if Imput_Cat == "Mode" :
            Change = str(Data[col].mode()[0])
        else:  
            Change = str(Imput_Cat)
        Data[col] = Data[col].fillna(Change)
    return(Data)   


def Data_X_y(features , Data , label) :
    Train = Data[features]
    if label in Data.columns :
        label = Data[label]
        return Train, label
        
    return Train



def encoder_colonnes_categorielles(df, colonnes, colonne_a_supprimer=None):
    # Supprimer la colonne spécifiée si elle existe
    if colonne_a_supprimer and colonne_a_supprimer in df.columns:
        df = df.drop(columns=[colonne_a_supprimer])
    
    # Encodage en 0/1 SANS drop_first
    X_encoded = pd.get_dummies(df[colonnes], drop_first=False).astype(int)
    
    print(f"✅ Encodage terminé. Dimensions : {df[colonnes].shape} → {X_encoded.shape}")
    return X_encoded


####### mon modèle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

def My_model(X, y, train_size=0.7, random_state=42):
   

    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )

    # 2. Entraînement du modèle
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    # 3. Probabilités prédites
    y_prob = model.predict_proba(X_test)[:, 1]

    #4 Calcule de seuil
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


    fp_weight = 2.0  # Poids fort sur les FP
    fn_weight = 1.0  # Poids standard sur les FN

    scores = fp_weight * fpr + fn_weight * (1 - tpr)  # 1 - tpr = taux de FN
    ix = np.argmin(scores)  # On cherche à minimiser le score
    best_thresh = thresholds[ix]

    # 5. Prédictions avec le seuil optimal
    y_pred_opt = (y_prob >= best_thresh).astype(int)

    return model, X_train, X_test, y_train, y_test, y_pred_opt, y_prob, best_thresh



import matplotlib.pyplot as plt
from sklearn import metrics

def evaluer_modele(modele, X_train, y_train, X_test, y_test,  y_pred_opt, y_prob,best_thresh):
    
    # Scores
    print(f"Score sur l'apprentisage : {modele.score(X_train, y_train):.2%}")
    print(f"Score sur le test : {modele.score(X_test, y_test):.2%}")
    
    
    # Log Loss (nécessite les probabilités prédites)
    print(f"Log Loss : {metrics.log_loss(y_test, y_prob)}")

    
    # 1. Courbe ROC
    
    plt.figure(figsize=(12, 5))
    
    
    roc_display = metrics.RocCurveDisplay.from_estimator(modele, X_test, y_test)
    plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale en noir
    plt.title('Courbe ROC')
    plt.legend()
    plt.show()
    
    # 2. Matrice de confusion VISUELLE
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred_opt,
        display_labels=["insatisfait", "satisfait"],  # Étiquettes personnalisables
        cmap='Blues',
        colorbar=False
    )
    plt.title('Matrice de Confusion (réelle vs Prédit)')
    plt.show()


    metric={'seuil_optimal': best_thresh,
            'accuracy': metrics.accuracy_score(y_test, y_pred_opt)}
   
    
    # Affichage des métriques
    print(f"\nSeuil optimal = {metric['seuil_optimal']:.4f}")
    print(f"proportion de bonnes prédictions(accuracy): {metric['accuracy']:.2%}")
    
    
    

    
    
   