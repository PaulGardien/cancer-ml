import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_overfitting_curve(X_train, X_test, y_train, y_test):
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    train_scores = []
    test_scores = []

    for C in C_values:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=C, max_iter=1000))
        ])
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(C_values, train_scores, 'o-', color='steelblue',
                label='Score train', linewidth=2)
    ax.semilogx(C_values, test_scores, 'o-', color='coral',
                label='Score test', linewidth=2)

    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.7,
               label='C=1 (sweet spot)')

    ax.set_xlabel('Valeur de C (échelle log)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting vs Underfitting selon C')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annoter les zones
    ax.text(0.001, min(train_scores) - 0.005,
            'Underfitting', fontsize=10, color='gray')
    ax.text(100, min(train_scores) - 0.005,
            'Overfitting', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('overfitting_curve.png', dpi=150)
    plt.close()
    print("Courbe sauvegardée : overfitting_curve.png")
    print(f"\n{'C':>10} | {'Train':>8} | {'Test':>8} | {'Gap':>8}")
    print("-" * 42)
    for C, tr, te in zip(C_values, train_scores, test_scores):
        gap = tr - te
        print(f"{C:>10.3f} | {tr:>8.4f} | {te:>8.4f} | {gap:>8.4f}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    plot_overfitting_curve(X_train, X_test, y_train, y_test)