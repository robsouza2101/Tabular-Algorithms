from utils.data_utils import load_dataset, preprocess_data
from utils.model_utils import get_models, evaluate_model
from utils.plot_utils import plot_results

# ================================
# 1. Carregar dataset
# ================================
X, y, feature_names, target_names = load_dataset("iris")

# ================================
# 2. Pré-processamento
# ================================
X_train, X_test, y_train, y_test = preprocess_data(X, y, scale=True)

# ================================
# 3. Definir modelos
# ================================
models = get_models(task="classification")

# ================================
# 4. Avaliar todos
# ================================
results = {}
for name, model in models.items():
    print(f"Treinando {name}...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, task="classification")
    results[name] = metrics

# ================================
# 5. Comparar
# ================================
df_results = plot_results(results, task="classification")
print("\nResumo Comparativo:")
print(df_results)
