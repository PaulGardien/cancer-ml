import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Recréer les données — même random_state=42 que l'entraînement
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=72, stratify=y
)

model = joblib.load(r"C:\work\Nightwork\cancer-ml\models\cancer_classifier.joblib")

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Seuil optimal : maximise TPR - FPR
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_threshold(model, X_test, y_test, threshold):
    """
    Au lieu d'utiliser model.predict() qui applique le seuil 0.5 par défaut,
    on récupère les probabilités et on applique notre propre seuil.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Appliquer le seuil manuellement
    y_pred = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    recall_malin = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    print(f"Seuil {threshold:.2f} | Precision: {precision:.3f} | "
          f"Recall bénin: {recall:.3f} | Recall malin: {recall_malin:.3f} | F1: {f1:.3f}")

if __name__ == '__main__':
    # ... charge données, entraîne modèle ...
    
    print("=== Impact du seuil de décision ===")
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        evaluate_threshold(model, X_test, y_test, threshold)