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

from sklearn.model_selection import cross_val_score
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("=== Impact du paramètre C (régularisation) ===")
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=C, max_iter=1000))
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"C={C:8.3f} | AUC = {scores.mean():.4f} ± {scores.std():.4f}")