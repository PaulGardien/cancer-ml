import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd


MODEL_PATH = "models/cancer_classifier.joblib"


def train_and_save():
    # Crée le dossier models/ si besoin
    os.makedirs("models", exist_ok=True)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # Sauvegarde
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    return X_test, y_test


def load_and_predict(X_test, y_test):
    # Chargement
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis : {MODEL_PATH}")

    # Prédiction sur les 5 premiers exemples
    sample = X_test.iloc[:5]
    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)[:, 1]

    labels = {0: 'Malin', 1: 'Bénin'}
    print("\n--- Prédictions sur 5 exemples ---")
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        print(f"Exemple {i+1} : {labels[pred]} (confiance bénin : {proba:.2%})")


if __name__ == '__main__':
    X_test, y_test = train_and_save()
    load_and_predict(X_test, y_test)