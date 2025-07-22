# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original : https://AILAB-CEFET-RJ/falldetection

Baseado no artigo (não publicado) - A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, K-Fold Cross Validation e o modelo LSTM foi implementado e testado junto aos demais.

Os sensores no trabalho original não são combinados (left, chest, right), por motivos de "escolher o melhor".

Suporta 3 arquiteturas de redes neurais, **CNN1D**, **MLP** e **LSTM**, com **otimização bayesiana de hiperparâmetros via Optuna**.

Também foi implementado o Early Stopping e o Median Pruning do Optuna.

## Funcionalidades

- **3 Arquiteturas de Redes Neurais**: CNN1D, MLP, LSTM
- **Otimização Bayesiana**: Via Optuna com Early Stopping
- **Análise de Importância**: Permutation Feature Importance
- **Explicabilidade**: Análise SHAP para interpretabilidade
- **Curvas de Aprendizado**: Análise de performance vs dados
- **Validação Cruzada**: K-Fold Cross Validation
- **Pipeline Automatizado**: Script `all.sh` para execução completa
- **Análise Global**: Comparações entre modelos e cenários

## Instalação e Configuração

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Baixe e Descompacte os Dados Originais
Disponível em: https://zenodo.org/records/12760391

### 3. Gere os Datasets e Labels
```bash
python generate_datasets.py chest
python generate_datasets.py left
python generate_datasets.py right
```

### 4. Valide os Datasets
```bash
python validate_datasets.py
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

### Pipeline Automatizado Completo

```bash
# Executar pipeline completo para múltiplas configurações
bash all.sh
```

O script `all.sh` executa automaticamente:
1. Validação dos datasets
2. Busca de hiperparâmetros (se necessário)
3. Treinamento final (se necessário)
4. Permutation Feature Importance
5. Curvas de aprendizado
6. Análise SHAP
7. Análise global

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
- **Sc_4_T**: Acelerômetro + Giroscópio 3 eixos, domínio temporal (Recomendado)
- **Sc_4_F**: Acelerômetro + Giroscópio 3 eixos, domínio frequência

> **Nota**: Sc_4_T é o cenário mais informativo e recomendado para análise.

O mais indicado seria um cenário 5, com:
- **Sc_5_T**: Acelerômetro + Giroscópio 3 eixos, + Magnitudes, domínio temporal

## Posições

- **chest**: Dados do peito (1020 samples) (Recomendado)
- **left**: Dados do lado esquerdo (450 samples)
- **right**: Dados do lado direito (450 samples)

## Tipos de Labels

- **binary_one**: Classificação binária (2 classes)
- **binary_two**: Classificação binária alternativa (2 classes) (Recomendado)

## Modelos de Rede Neural

- **CNN1D**: Rede neural convolucional 1D
- **MLP**: Multi-layer perceptron
- **LSTM**: Long short-term memory

## Estrutura do Projeto

```
fall-detect/
├── config.py                 # Configurações centralizadas
├── hyperparameter_search.py  # Script para busca de hiperparâmetros
├── post_trials.py           # Processamento pós-trials do Optuna
├── final_training.py         # Script para treinamento final + análise
├── utils.py                  # Funções utilitárias organizadas
├── neural_networks.py        # Arquiteturas das redes neurais
├── requirements.txt          # Dependências
├── all.sh                    # Pipeline automatizado completo
├── analysis.py               # Análise global dos resultados
├── learning_curve.py         # Geração de curva de aprendizado
├── permutation_importance.py # Permutation Feature Importance
├── shap_importance.py        # Análise SHAP para explicabilidade
├── validate_datasets.py      # Validação dos datasets gerados
├── generate_datasets.py      # Geração de datasets
├── builders/                 # Builders para dados
├── labels_and_data/          # Dados e labels
├── database/                 # Base de dados
├── analise_global/           # Resultados agregados e gráficos
└── README.md                 # Este arquivo
```

## Saídas Geradas

### Busca de Hiperparâmetros
- `best_hyperparameters.json`: Melhores hiperparâmetros encontrados
- `test_data.npz`: Dados de treino/validação e teste salvos
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
  - `best_models/`: Diretório com cópias dos melhores modelos

### Permutation Importance
- `permutation_importance.csv`: Importância das features via permutação
- `permutation_importance.png`: Gráfico de importância das features

### Curva de Aprendizado
- `learning_curve.csv`: Dados da curva de aprendizado
- `learning_curve_metrics.csv`: Métricas por fração de dados
- `learning_curve.png`: Gráfico da curva de aprendizado

### Análise SHAP
- `shap_values_*.npy`: Valores SHAP salvos
- `shap_importance_*.csv`: Importância das features via SHAP
- `shap_importance_*.png`: Gráficos de importância SHAP
- `shap_importance_class*_*.csv/png`: Análise por classe

### Análise Global (`analise_global/`)
- **Boxplots**: Comparações entre modelos e métricas
- **Curvas ROC**: Comparações de performance
- **Matrizes de Confusão**: Agregadas por modelo
- **Curvas de Aprendizado**: Comparações de learning curves
- **Importância de Features**: Permutation e SHAP
- **Análise Optuna**: Convergência e importância de parâmetros
- **Relatórios de Classificação**: Métricas detalhadas

Aqui, focamos em gerar o melhor modelo, são dispositivos diferentes com frequências diferentes, além de que precisariamos das 3 entradas para funcionar, então foi usado para análise o dataset do peito.

## Exemplos de Uso

### Execução Rápida (Recomendado)
```bash
# Pipeline completo para o cenário mais informativo
bash all.sh
```

### Execução Personalizada
```bash
# 1. Validar datasets
python validate_datasets.py

# 2. Buscar hiperparâmetros para LSTM
python hyperparameter_search.py -scenario Sc_4_T -position chest -label_type binary_two --nn LSTM

# 3. Treinar modelos finais
python final_training.py -scenario Sc_4_T -position chest -label_type binary_two --nn LSTM --num_models 20

# 4. Análise de importância
python permutation_importance.py -scenario Sc_4_T -position chest -label_type binary_two --nn LSTM

# 5. Curva de aprendizado
python learning_curve.py -scenario Sc_4_T -position chest -label_type binary_two --nn LSTM

# 6. Análise SHAP
python shap_importance.py -scenario Sc_4_T -position chest -label_type binary_two --nn LSTM

# 7. Análise global
python analysis.py
```
