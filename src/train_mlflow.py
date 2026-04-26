import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score
)
import numpy as np


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_log(model_name, model, X_train, X_test, y_train, y_test, params):
    """Entraîne un modèle et logge tout dans MLflow."""

    with mlflow.start_run(run_name=model_name):

        # 1. Logger les paramètres
        mlflow.log_params(params)

        # 2. Entraîner
        model.fit(X_train, y_train)

        # 3. Calculer les métriques
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision_malin": precision_score(y_test, y_pred, pos_label=0),
            "recall_malin": recall_score(y_test, y_pred, pos_label=0),
        }

        # Cross-validation AUC
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics["cv_auc_mean"] = cv_scores.mean()
        metrics["cv_auc_std"] = cv_scores.std()

        # 4. Logger les métriques
        mlflow.log_metrics(metrics)

        # 5. Logger le modèle
        mlflow.sklearn.log_model(model, "model")

        # Affichage
        print(f"\n=== {model_name} ===")
        for k, v in metrics.items():
            print(f"  {k:25s} : {v:.4f}")

        return metrics


if __name__ == '__main__':

    mlflow.set_experiment("cancer-classification")
    X_train, X_test, y_train, y_test = load_data()

    # Run 1 — Régression Logistique
    train_and_log(
        model_name="LogisticRegression",
        model=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ]),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={"model": "LogisticRegression", "C": 1.0, "max_iter": 1000}
    )

    # Run 2 — Random Forest
    train_and_log(
        model_name="RandomForest_100",
        model=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={"model": "RandomForest", "n_estimators": 100}
    )

    # Run 3 — Random Forest avec plus d'arbres
    train_and_log(
        model_name="RandomForest_200",
        model=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={"model": "RandomForest", "n_estimators": 200}
    )

    print("\n✓ Tous les runs sont loggés.")
    print("Lance maintenant : mlflow ui")
    print("Puis ouvre : http://localhost:5000")