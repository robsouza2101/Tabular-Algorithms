import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_classification_from_csv(csv_path="results/model_metrics.csv", save_path=None):
    """
    Lê métricas de classificação de um CSV e gera gráficos comparativos.

    Parameters
    ----------
    csv_path : str
        Caminho para o arquivo CSV com métricas.
    save_path : str
        Caminho opcional para salvar o gráfico (PNG).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado.")

    df = pd.read_csv(csv_path).set_index("Modelo")

    # Seleciona apenas as métricas principais
    metrics = ["Acurácia", "Precisão (Macro)", "Recall (Macro)", "F1-score (Macro)"]
    df_metrics = df[metrics]

    # --- Gráfico comparativo
    plt.figure(figsize=(12,6))
    df_metrics.plot(kind="bar", figsize=(12,6), rot=45)
    plt.title("Comparação de Modelos - Métricas de Classificação")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Gráfico salvo em {save_path}")
    plt.show()

    return df


def plot_regression_from_csv(csv_path="results/regression_metrics.csv", save_path=None):
    """
    Lê métricas de regressão de um CSV e gera gráficos comparativos.

    Parameters
    ----------
    csv_path : str
        Caminho para o arquivo CSV com métricas.
    save_path : str
        Caminho opcional para salvar os gráficos (PNG).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado.")

    df = pd.read_csv(csv_path).set_index("Modelo")

    # --- Gráfico R²
    plt.figure(figsize=(10,6))
    df["R²"].plot(kind="bar", rot=45, color="skyblue")
    plt.title("Comparação de Modelos - R²")
    plt.ylabel("R²")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path.replace(".png", "_r2.png"), bbox_inches="tight")
        print(f"Gráfico R² salvo em {save_path.replace('.png', '_r2.png')}")
    plt.show()

    # --- Gráfico de erros
    df_err = df[["RMSE", "MAE"]]
    df_err.plot(kind="bar", figsize=(10,6), rot=45)
    plt.title("Comparação de Modelos - Erros (RMSE e MAE)")
    plt.ylabel("Erro")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path.replace(".png", "_errors.png"), bbox_inches="tight")
        print(f"Gráfico de erros salvo em {save_path.replace('.png', '_errors.png')}")
    plt.show()

    return df

if __name__ == "__main__":
    # Exemplo de uso
    plot_classification_from_csv(csv_path="results/model_metrics.csv", save_path="results/classification_comparison.png")
    # plot_regression_from_csv(csv_path="results/regression_metrics.csv", save_path="results/regression_comparison.png")