# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original : https://AILAB-CEFET-RJ/falldetection

Baseado no artigo - A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, K-Fold Cross Validation e o modelo LSTM foi implementado e testado junto aos demais.

Os sensores no trabalho original não são combinados (left, chest, right), por motivos de "escolher o melhor".

Suporta 3 arquiteturas de redes neurais, **CNN1D**, **MLP** e **LSTM**, com **otimização bayesiana de hiperparâmetros via Optuna**.

Também foi implementado o Early Stopping e o Median Pruning do Optuna.

## Estrutura do Projeto

```
fall-detect/
├── config.py                 # Configurações centralizadas
├── hyperparameter_search.py  # Script para busca de hiperparâmetros
├── final_training.py         # Script para treinamento final + análise
├── utils.py                  # Funções utilitárias organizadas
├── neural_networks.py        # Arquiteturas das redes neurais
├── requirements.txt          # Dependências
├── all.sh                    # Pipeline automatizado
├── analysis.py               # Análise global dos resultados
├── learning_curve.py         # Geração de curva de aprendizado
├── permutation_importance.py # Permutation Feature Importance
├── generate_datasets.py      # Geração de datasets
├── builders/                 # Builders para dados
├── labels_and_data/          # Dados e labels
├── database/                 # Base de dados
├── analise_global/           # Resultados agregados e gráficos
└── README.md                 # Este arquivo
```

## Início

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 2. Baixe e Descompacte os Dados Originais
Disponível em: https://zenodo.org/records/12760391

### 3. Gere os Datasets e Labels
```bash
python generate_datasets.py chest
python generate_datasets.py left
python generate_datasets.py right
```

## Configurações

O arquivo `config.py` centraliza todas as configurações do projeto:

- **Dispositivo**: Configuração automática de GPU/CPU
- **Seeds**: Reprodutibilidade dos experimentos
- **Diretórios**: Caminhos para dados e saídas
- **Cenários**: Configurações dos diferentes cenários de dados
- **Hiperparâmetros**: Ranges para otimização
- **Treinamento**: Configurações de treinamento

## Modos de Execução

### 1. Busca de Hiperparâmetros (`hyperparameter_search.py`)

```bash
python hyperparameter_search.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D --n_trials 20 --timeout 3600
```

### 2. Treinamento Final (`final_training.py`)

```bash
python final_training.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D --num_models 20 --epochs 25
```

**Nota:** A análise de resultados é executada automaticamente após o treinamento. Use `--no_analysis` para pular a análise.

### 3. Permutation Feature Importance (`permutation_importance.py`)

```bash
python permutation_importance.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D
```

### 4. Curva de Aprendizado (`learning_curve.py`)

```bash
python learning_curve.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D
```

### 5. Análise Global dos Resultados (`analysis.py`)

```bash
python analysis.py
```

### 6. Pipeline Automatizado

```bash
# Executar pipeline completo para múltiplas configurações
bash all.sh
```

## Fluxo de Trabalho Recomendado

1. **Busca de Hiperparâmetros**
   ```bash
   python hyperparameter_search.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D --n_trials 50
   ```
2. **Treinamento Final (com análise automática)**
   ```bash
   python final_training.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D --num_models 20
   ```
3. **Permutation Importance**
   ```bash
   python permutation_importance.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D
   ```
4. **Curva de Aprendizado**
   ```bash
   python learning_curve.py -scenario Sc1_acc_T -position chest -label_type binary_one --nn CNN1D
   ```
5. **Análise Global**
   ```bash
   python analysis.py
   ```
6. **Pipeline Automatizado**
   ```bash
   bash all.sh
   ```

## Cenários Disponíveis

- **Sc1_acc_T**: Acelerômetro magnitude, domínio temporal
- **Sc1_gyr_T**: Giroscópio magnitude, domínio temporal
- **Sc1_acc_F**: Acelerômetro magnitude, domínio frequência
- **Sc1_gyr_F**: Giroscópio magnitude, domínio frequência
- **Sc_2_acc_T**: Acelerômetro 3 eixos, domínio temporal
- **Sc_2_gyr_T**: Giroscópio 3 eixos, domínio temporal
- **Sc_2_acc_F**: Acelerômetro 3 eixos, domínio frequência
- **Sc_2_gyr_F**: Giroscópio 3 eixos, domínio frequência
- **Sc_3_T**: Acelerômetro + Giroscópio magnitude, domínio temporal
- **Sc_3_F**: Acelerômetro + Giroscópio magnitude, domínio frequência
- **Sc_4_T**: Acelerômetro + Giroscópio 3 eixos, domínio temporal
- **Sc_4_F**: Acelerômetro + Giroscópio 3 eixos, domínio frequência

(Faltou o SC_5 magnitude + 3 eixos)
Na prática, para a análise deste trabalho, não existe necessidade de testar todos os cenários, sabemos que em deeplearning, podemos pegar o dataset mais informativo sem problemas.

## Posições

- **chest**: Dados do peito (1020 samples)
- **left**: Dados do lado esquerdo (450 samples)
- **right**: Dados do lado direito (450 samples)

## Tipos de Labels

- **binary_one**: Classificação binária (2 classes)
- **binary_two**: Classificação binária alternativa (2 classes)
- **multiple_one**: Classificação múltipla (37 classes)
- **multiple_two**: Classificação múltipla alternativa (26 classes)

## Modelos de Rede Neural

- **CNN1D**: Rede neural convolucional 1D
- **MLP**: Multi-layer perceptron
- **LSTM**: Long short-term memory

## Saídas Geradas

### Busca de Hiperparâmetros
- `best_hyperparameters.json`: Melhores hiperparâmetros encontrados
- `test_data.npz`: Dados de teste salvos
- `optuna_trials.csv`: Resultados de todos os trials
- `param_importance.png`/`.html`: Importância dos hiperparâmetros
- Diretórios `trial_X/`: Resultados de cada trial do Optuna

### Treinamento Final + Análise
- Diretórios `model_X/`: Resultados de cada modelo treinado
  - `model_X.pt`: Modelo salvo
  - `metrics_model_X.csv`/`.txt`: Métricas do modelo
  - `loss_curve_model_X.png`: Curva de loss
  - `confusion_matrix_model_X.png`: Matriz de confusão
  - `roc_curve_model_X.png`: Curva ROC
  - `classification_report_model_X.txt`: Relatório de classificação
- **Análise automática:**
  - `all_metrics.csv`: Métricas de todos os modelos
  - `summary_metrics.csv`: Estatísticas resumidas
  - `metrics_boxplot.png`: Boxplot das métricas
  - `mcc_histogram.png`: Histograma do MCC
  - `mcc_vs_accuracy.png`: Scatter plot MCC vs Accuracy
  - `correlation_heatmap.png`: Matriz de correlação
  - `best_models/`: Diretório com cópias dos melhores modelos

### Permutation Importance
- `permutation_importance.csv`: Importância das features via permutação
- `permutation_importance.png`: Gráfico de importância das features

### Curva de Aprendizado
- `learning_curve.csv`: Dados da curva de aprendizado
- `learning_curve_metrics.csv`: Métricas por fração de dados
- `learning_curve.png`: Gráfico da curva de aprendizado

### Análise Global (analise_global/)
- Boxplots, curvas, gráficos agregados de todos os experimentos
- Comparações entre modelos, cenários, posições e métricas

### Modificar Configurações

Edite o arquivo `config.py` para alterar:

- Número de trials do Optuna
- Configurações de treinamento
- Ranges de hiperparâmetros
- Configurações de GPU