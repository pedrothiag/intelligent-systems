# 🤖 Sistemas Inteligentes

Neste repositório estão os materiais de suporte e códigos desenvolvidos para a disciplina de Sistemas Inteligentes, ministrada no semestre de 2026.2 na UFERSA - Campus Pau dos Ferros.

## 📅 Assuntos Abordados no Curso

| # | Nome | Slides | Notebook |
|---|---|---|---|
| 1 | Introdução à Inteligência Artificial | [Slides](./Slides/01_introducao_a_inteligencia_artificial.pdf) | |
| 2 | Introdução ao Aprendizado de Máquina | [Slides](./Slides/02_aprendizado_maquina.pdf) | |
| 3 | Análise Exploratória de Dados | [Slides](./Slides/03_analise_dados.pdf) | [Notebook](./Notebooks/03_Analise_Exploratoria_Dados.ipynb) |
| 4 | Pré-Processamento de Dados | [Slides](./Slides/04_pre_processamento.pdf) | [Notebook](./Notebooks/04_Preprocessamento.ipynb) |
| 5 | Métricas de Performance | [Slides](./Slides/05_metricas_performance.pdf) | |
| 6 | Regressão Linear | [Slides](./Slides/06_regressao_linear.pdf) | [Notebook](./Notebooks/05_Regressao_Linear.ipynb) |
| 7 | Regressão Logística | [Slides](./Slides/07_regressao_logistica.pdf) | [Notebook](./Notebooks/06_Regressao_Logistica.ipynb) |
| 8 | k-Nearest Neighbors (kNN) | [Slides](./Slides/08_knn.pdf) | [Notebook](./Notebooks/07_kNN.ipynb) |
| 9 | Árvores de Decisão | [Slides](./Slides/09_arvores_de_decisao.pdf) | [Notebook](./Notebooks/08_Tree.ipynb) |
| 10 | K-Means | [Slides](./Slides/10_kmeans.pdf) | [Notebook](./Notebooks/09_Clusterizacao.ipynb) |
| 11 | Introdução às Redes Neurais Artificiais | [Slides](./Slides/11_introducao_rna.pdf) | |
| 12 | Perceptron | [Slides](./Slides/12_perceptron.pdf) | [Notebook](./Notebooks/10_Percepetron.ipynb) |
| 13 | Adaline | [Slides](./Slides/13_adaline.pdf) | [Notebook](./Notebooks/11_Adaline.ipynb) |
| 14 | Redes Neurais Multicamadas (MLP) | [Slides](./Slides/14_mlp.pdf) | [Notebook](./Notebooks/12_Redes_MLP.ipynb) |
| 15 | Redes Neurais Convolucionais (CNN) | [Slides](./Slides/15_cnn.pdf) | [Notebook](./Notebooks/13_CNN.ipynb) |

## 📦 Instalação das Bibliotecas

Para utilizar os códigos desenvolvidos neste curso, recomenda-se:

- Criar um ambiente virtual para o desenvolvimento;
- Se você tiver placas da nVIDIA, instale o CUDA [[Link]](https://developer.nvidia.com/cuda-downloads) e, após isso, execute os seguites comandos:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy scipy pandas matplotlib scikit-learn opencv-python notebook seaborn imblearn
```

- Caso você não tenha placas da nVIDIA, basta instalar os seguintes pacotes:

```
pip3 install torch torchvision
pip install numpy scipy pandas matplotlib scikit-learn opencv-python notebook seaborn imblearn
```

## 📝 Lista de Exercícios e Projetos

| Unidade | Tópicos abordados | Lista de exercícios | Lista de projetos |
|---|---|---|---|
| 1 | Introdução à Inteligência Artificial<br>Introdução ao Aprendizado de Máquina<br>Análise Exploratória de Dados<br>Pré-Processamento de Dados<br>Métricas de Performance<br>Regressão Linear<br>Regressão Logística<br>k-Nearest Neighbors (kNN)<br>Árvores de Decisão<br>K-Means | [Lista de Exercícios — Unidade 1](./Listas/01_lista_unidade1.pdf) | [Lista de Projetos — Unidade 1](./Projetos/01_lista_1_projetos.pdf) |
