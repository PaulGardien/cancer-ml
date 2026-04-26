import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def search_logistic_regression(X_train, X_test, y_train, y_test):
    """GridSearch sur LogisticRegression."""

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # La grille : toutes les combinaisons à tester
    # Note : pour cibler un paramètre dans un pipeline,
    # on utilise la syntaxe "nom_etape__nom_parametre"
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__solver': ['lbfgs', 'liblinear']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,                    # 5 folds
        scoring='roc_auc',       # métrique d'optimisation
        n_jobs=-1,               # utilise tous les CPU
        verbose=1                # affiche la progression
    )

    grid_search.fit(X_train, y_train)

    # Meilleur modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    results = {
        'best_params': grid_search.best_params_,
        'best_cv_auc': grid_search.best_score_,
        'test_auc': roc_auc_score(y_test, y_proba),
        'recall_malin': recall_score(y_test, y_pred, pos_label=0),
        'precision_malin': precision_score(y_test, y_pred, pos_label=0)
    }

    return best_model, results


def log_to_mlflow(model, results, run_name):
    """Logger les résultats du GridSearch dans MLflow."""

    mlflow.set_experiment("cancer-gridsearch")

    with mlflow.start_run(run_name=run_name):
        # Logger les meilleurs paramètres trouvés
        mlflow.log_params(results['best_params'])

        # Logger les métriques
        mlflow.log_metrics({
            'best_cv_auc': results['best_cv_auc'],
            'test_auc': results['test_auc'],
            'recall_malin': results['recall_malin'],
            'precision_malin': results['precision_malin']
        })

        # Logger le modèle
        mlflow.sklearn.log_model(model, "model")

        print(f"\n=== {run_name} ===")
        print(f"Meilleurs paramètres : {results['best_params']}")
        print(f"CV AUC (best)        : {results['best_cv_auc']:.4f}")
        print(f"Test AUC             : {results['test_auc']:.4f}")
        print(f"Recall malin         : {results['recall_malin']:.4f}")
        print(f"Precision malin      : {results['precision_malin']:.4f}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    best_model, results = search_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    log_to_mlflow(best_model, results, run_name="LR_GridSearch")

    print("\n✓ Lance : mlflow ui pour voir les résultats")