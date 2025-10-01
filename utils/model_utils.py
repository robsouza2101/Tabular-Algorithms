from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os

def get_models(task="classification"):
    """Get a dictionary of models for a given task.

    Args:
        task (str, optional): Wich task is going to be performed. Defaults to "classification".

    Returns:
        dict: A dictionary with model names as keys and model instances as values.
    """
    if task == "classification":
        models = {
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=500, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500)
        }
    else:  # regressão
        models = {
            "Linear Regression": LinearRegression()
        }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, task="classification"):
    """Evaluate a model and return relevant metrics.

    Args:
        model (_type_): model to be evaluated
        X_train (_type_): data for training
        X_test (_type_): data for testing
        y_train (_type_): target for training
        y_test (_type_): target for testing
        task (str, optional): Classification or regression. Defaults to "classification".

    Returns:
        _type_: metrics dictionary
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc, "report": classification_report(y_test, y_pred, output_dict=True)}
    
    elif task == "regression":
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        return {"R²": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

def save_classification_results(results, filename="results/model_metrics.csv"):
    """
    Salva métricas de múltiplos modelos em um CSV.

    Parameters
    ----------
    results : dict
        Dicionário no formato { "Modelo": {"accuracy": valor, "report": classification_report_dict}, ... }
    filename : str
        Caminho do arquivo CSV de saída
    """
    rows = []

    for model_name, metrics in results.items():
        acc = metrics["accuracy"]
        report = metrics["report"]

        # Macro avg
        precision_macro = report["macro avg"]["precision"]
        recall_macro = report["macro avg"]["recall"]
        f1_macro = report["macro avg"]["f1-score"]

        # Melhor classe (maior f1-score entre as classes)
        best_class = max([c for c in report.keys() if c not in ["accuracy","macro avg","weighted avg"]],
                         key=lambda c: report[c]["f1-score"])
        best_precision = report[best_class]["precision"]
        best_recall = report[best_class]["recall"]
        best_f1 = report[best_class]["f1-score"]

        # Montar linha
        rows.append({
            "Modelo": model_name,
            "Acurácia": acc,
            "Precisão (Macro)": precision_macro,
            "Recall (Macro)": recall_macro,
            "F1-score (Macro)": f1_macro,
            "Classe Melhor F1": best_class,
            "Precisão (Best)": best_precision,
            "Recall (Best)": best_recall,
            "F1-score (Best)": best_f1
        })

    # Converter para DataFrame
    df = pd.DataFrame(rows)

    # Salvar em CSV
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False, encoding="utf-8")

    print(f"Resultados salvos em {filename}")
    return df
    
if __name__ == "__main__":
    from data_utils import load_dataset, preprocess_data

    df = load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = preprocess_data(df.data, df.target, scale=True)
    models = get_models(task="classification")
    metrics = {}
    for name, model in models.items():
        print(f"\n\nEvaluating {name}...")
        metric = evaluate_model(model, X_train, X_test, y_train, y_test, task="classification")
        metrics[name] = metric
        print(f"Accuracy: {metric['accuracy']:.4f}")
        print("Classification Report:")
        print(metric['report'])
        save_classification_results(results=metrics)
