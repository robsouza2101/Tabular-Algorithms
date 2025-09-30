from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
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
    
if __name__ == "__main__":
    print(os.getcwd())
    from data_utils import load_dataset, preprocess_data

    df = load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = preprocess_data(df.data, df.target, scale=True)
    models = get_models(task="classification")
    for name, model in models.items():
        print(f"\n\nEvaluating {name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, task="classification")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification Report:")
        print(metrics['report'])
