import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score


def load_and_split():
    data = load_breast_cancer()
    import pandas as pd
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_and_train(X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    # La courbe ROC
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'Logistic Regression (AUC = {auc:.4f})')

    # La ligne aléatoire (baseline)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aléatoire (AUC = 0.5)')

    # Trouver le seuil optimal (maximise TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               color='red', zorder=5, s=100,
               label=f'Seuil optimal = {optimal_threshold:.2f}')

    ax.set_xlabel('Taux de Faux Positifs (FPR)')
    ax.set_ylabel('Taux de Vrais Positifs (TPR = Recall)')
    ax.set_title('Courbe ROC — Classification Cancer')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.close()
    print(f"Courbe ROC sauvegardée.")
    print(f"AUC : {auc:.4f}")
    print(f"Seuil optimal : {optimal_threshold:.4f}")
    print(f"  → TPR (recall) à ce seuil : {tpr[optimal_idx]:.4f}")
    print(f"  → FPR à ce seuil : {fpr[optimal_idx]:.4f}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split()
    model = build_and_train(X_train, y_train)
    plot_roc_curve(model, X_test, y_test)