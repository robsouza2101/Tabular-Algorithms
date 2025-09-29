# 📊 ML Tabular Algorithms

Este repositório reúne implementações práticas de **algoritmos clássicos de Machine Learning aplicados a dados tabulares**, com o objetivo de comparar o desempenho de diferentes modelos em um mesmo dataset.  

A ideia principal é construir uma base sólida de estudo em ML supervisionado, praticar **Git** no versionamento e criar um repositório organizado e reutilizável para futuras consultas.

---

## 🎯 Objetivos
- Implementar os principais algoritmos de ML para dados tabulares.  
- Avaliar e comparar o desempenho dos modelos em datasets de exemplo.  
- Criar um fluxo padronizado de **pré-processamento → treinamento → avaliação → comparação**.  
- Consolidar aprendizado em **Git e versionamento de código científico**.  

---

## 🏗️ Arquitetura do Projeto
O projeto foi estruturado de forma modular para facilitar a manutenção e a comparação entre modelos:

tabular-algorithms/
│── data/ # Datasets utilizados (ex: Iris, Titanic, Wine)
│── notebooks/ # Jupyter Notebooks de cada algoritmo
│ ├── naive_bayes.ipynb
│ ├── decision_tree.ipynb
│ ├── random_forest.ipynb
│ ├── linear_regression.ipynb
│ ├── knn.ipynb
│ ├── svm.ipynb
│ └── mlp.ipynb
│── results/ # Métricas e comparações salvas
│── utils/ # Funções auxiliares (ex: métricas, gráficos)
│── compare_models.ipynb # Notebook para análise comparativa final
│── README.md # Documentação do projeto


Fluxo de execução (padrão para todos os modelos):  
1. **Carregamento dos dados** → (datasets do `scikit-learn` ou UCI).  
2. **Pré-processamento** → tratamento de missing values, normalização, codificação de variáveis.  
3. **Treinamento** → uso de `scikit-learn` (ou Keras/PyTorch para MLP).  
4. **Avaliação** → métricas (acurácia, precisão, recall, F1, AUC) e matriz de confusão.  
5. **Comparação** → resultados salvos em CSV/JSON e gráficos consolidados.  

---

## 🤖 Algoritmos Implementados
- **Naive Bayes** (classificação probabilística baseada no Teorema de Bayes).  
- **Árvore de Decisão** (modelo interpretável de regras).  
- **Random Forest** (ensemble de árvores, robusto contra overfitting).  
- **Regressão Linear** (modelo base para regressão).  
- **k-Nearest Neighbors (kNN)** (classificação baseada em distâncias).  
- **Support Vector Machine (SVM)** (ótimo para fronteiras complexas).  
- **Multilayer Perceptron (MLP)** (rede neural simples aplicada a tabulares).  

---

## 📈 Resultados Esperados
- Tabelas comparativas de métricas.  
- Gráficos de acurácia/F1-score para análise de desempenho.  
- Insights sobre pontos fortes e fracos de cada algoritmo em dados tabulares.  

---

## 🛠️ Tecnologias Utilizadas
- **Python 3**  
- **scikit-learn** (modelos clássicos)  
- **matplotlib / seaborn** (visualização)  
- **pandas / numpy** (manipulação de dados)  
- **Jupyter Notebook** (ambiente interativo)  
- **Git & GitHub** (versionamento e organização)  

---

## 🚀 Como Usar
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/ml-tabular-algorithms.git
   cd ml-tabular-algorithms
   ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Acesse os notebooks:
    ```bash
    jupyter notebook
    ```

Execute os experimentos e veja os resultados em results/.