import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt

def see_data_info(data):
    """Display basic information about a dataset.

    Parameters:
    data: sklearn dataset (ex: load_iris())
    """
    # Montar DataFrame com features
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # Adicionar coluna target
    df["target"] = data.target  

    print("Dimensões do dataset:", df.shape)
    print("\nPrimeiras linhas:")
    print(df.head())  
    
    print("\nResumo estatístico:")
    print(df.describe())

    print("\nDistribuição das classes:")
    print(df["target"].value_counts().rename(index=dict(enumerate(data.target_names))))

    print("\nInformações do DataFrame:")
    print(df.info())



def load_dataset(name="iris"):
    """Load a dataset by name.

    Parameters:
    name (str): The name of the dataset to load. Options are 'iris', 'wine', 'breast_cancer'.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    
    return data

if __name__ == "__main__":
    # Example usage
    df = load_dataset("breast_cancer")
    see_data_info(df)