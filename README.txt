<div align="center">

# Prédiction de la Satisfaction des Passagers Aériens

<img src="https://img.icons8.com/clouds/200/airplane-take-off.png" alt="Avion" width="150"/>

### Projet Master 1 Data Science – UFHB 2025-2026

**KOFFI KOUAME Jean Baptiste**  
**KOUASSI KOUADIO Prosper**

**Random Forest – Accuracy 96.35 %** • **Seuil optimal : 0.5773**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-2965C5)](https://pandas.pydata.org)
[![Scikit--Learn](https://img.shields.io/badge/Scikit_Learn-1.5-F7931E)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Statut-Terminé-brightgreen?style=for-the-badge)](https://github.com/JBKOFFI/projet-satisfaction-aerienne)

</div>

## Contexte & Objectif du projet

Cet ensemble de données provient d’une enquête de satisfaction auprès des passagers d’une compagnie aérienne.

**Question métier** :  
Quels facteurs influencent le plus la satisfaction des passagers ?  
**Objectif technique** :  
Construire le **meilleur modèle de scoring** capable de prédire si un passager sera **Satisfait** ou **Insatisfait/Neutre**, avec un seuil de décision optimisé.

## Description des données (24 variables)

| Catégorie               | Variables principales |
|-------------------------|-----------------------|
| Profil passager         | Sexe, Âge, Type de client (fidèle/infidéle) |
| Voyage                  | Type de voyage, Classe (Eco, Eco Plus, Business), Distance |
| Services notés (0–5)    | Wi-Fi, Restauration, Confort siège, Divertissement, Propreté, Service en vol… |
| Ponctualité             | Retard départ, Retard arrivée (en minutes) |
| Variable cible          | **Satisfaction** → `Satisfied` / `Neutral or Dissatisfied` (regroupé en **Insatisfait**) |

Dataset : ~130 000 passagers

## Travail réalisé

| Étape                        | Réalisé |
|-----------------------------|----------|
| Chargement & nettoyage      | Oui (feuille "Data" du fichier Excel) |
| Analyse exploratoire (EDA)  | Oui (graphiques + corrélations) |
| Feature engineering         | Oui (gestion des valeurs manquantes, encodage) |
| Modélisation                | Oui (Logistic Regression, Random Forest, XGBoost, LightGBM) |
| Optimisation du seuil       | Oui (Youden’s J → seuil = **0.5773**) |
| Interprétation métier       | Oui |

## Résultats du modèle retenu – Random Forest

| Métrique                     | Score          |
|------------------------------|----------------|
| Accuracy (test)              | **96.35 %**    |
| Accuracy globale             | 96.27 %        |
| AUC-ROC                      | 0.994          |
| Nombre d’erreurs (test)      | 999 / ~33 500  |
| Seuil de décision optimal    | **0.5773**     |

→ Le modèle prédit correctement **96.65 %** des cas avec un seuil ajusté

## Contenu du repository
