import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')  # désactive l'affichage interactif, sauvegarde uniquement
import matplotlib.pyplot as plt


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y


def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),       # normalisation
        ('classifier', LogisticRegression(max_iter=1000))
    ])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                                target_names=['Malin', 'Bénin']))

    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Malin', 'Bénin'],
        cmap='Blues'
    )
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.savefig('confusion_matrix.png')
plt.close()
print("Matrice de confusion sauvegardée : confusion_matrix.png")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def compare_models(X, y):
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    print("\n=== Comparaison par cross-validation (5 folds) ===")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"{name:25s} AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")


if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)
    compare_models(X, y)