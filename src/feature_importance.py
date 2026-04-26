import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return (X,
            train_test_split(X, y, test_size=0.2,
                             random_state=42, stratify=y))


def plot_feature_importance(X, X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)

    # Extraire les importances
    importances = model['clf'].feature_importances_
    feature_names = X.columns

    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("=== Top 10 features les plus importantes ===")
    print(df_imp.head(10).to_string(index=False))

    # Plot top 15
    fig, ax = plt.subplots(figsize=(10, 8))
    df_top = df_imp.head(15)
    ax.barh(df_top['feature'][::-1], df_top['importance'][::-1],
            color='steelblue')
    ax.set_xlabel('Importance (Gini)')
    ax.set_title('Top 15 features — Random Forest')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("\nGraphique sauvegardé : feature_importance.png")


if __name__ == '__main__':
    X, (X_train, X_test, y_train, y_test) = load_data()
    plot_feature_importance(X, X_train, y_train)