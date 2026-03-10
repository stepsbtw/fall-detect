# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original: https://github.com/AILAB-CEFET-RJ/falldetection

Baseado no artigo (preprint) — A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, **Leave-One-Group-Out (LOGO) Cross Validation**, o modelo LSTM e a fusão de sensores foram implementados e testados junto aos demais.

Suporta 3 arquiteturas de redes neurais, **CNN1D**, **MLP** e **LSTM**, com **otimização bayesiana de hiperparâmetros via Optuna**.

## Funcionalidades

- **3 Arquiteturas de Redes Neurais**: CNN1D, MLP, LSTM
- **Otimização Bayesiana**: Via Optuna com Early Stopping e Median Pruning
- **6 Posições de Sensor**: individuais (chest, left, right) e fusões (chest\_left, chest\_right, chest\_left\_right)
- **Domínio Temporal e de Frequência**: todos os cenários disponíveis nos dois domínios
- **Validação Cruzada LOGO**: Leave-One-Group-Out por participante
- **Explicabilidade**: Análise SHAP
- **Curvas de Aprendizado**: Análise de performance vs. quantidade de dados
- **Análise Global**: Comparações entre modelos e cenários

## Instalação e Configuração

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Baixe e Descompacte os Dados Originais

Disponível em: https://doi.org/10.5281/zenodo.12760390

Extraia na pasta `dataset/0_raw/` de forma que a estrutura fique:
```
dataset/0_raw/ID1/CHEST/  ID1/LEFT/  ID1/RIGHT/
               ID2/...
               ...
               ID15/...
```

### 3. Gere os Datasets (sensores individuais)

```bash
cd src/data

# Gera os três sensores de uma vez
python generate_datasets.py

# Ou um sensor por vez
python generate_datasets.py chest
python generate_datasets.py left
python generate_datasets.py right
```

### 4. Gere os Datasets Fundidos (fusão de sensores)

```bash
# Todas as combinações de dois ou três sensores
python generate_fused_dataset.py --positions chest left
python generate_fused_dataset.py --positions chest right
python generate_fused_dataset.py --positions chest left right
```

### 5. Valide os Datasets

```bash
python validate_dataset.py
python validate_fused_dataset.py
```

## Configurações

O arquivo `src/config.py` centraliza todas as configurações:

- **Dispositivo**: Seleção automática GPU/CPU
- **Seed**: Reprodutibilidade (`SEED = 42`)
- **Cenários**: Mapeamento posição → arquivo → shape de entrada
- **Hiperparâmetros**: Ranges para otimização por arquitetura
- **Treinamento**: epochs, patience, batch\_size, etc.
- **Split de dados**: 80% treino/val — 20% teste

## Pipeline de Treinamento (`pipeline.py`)

Todos os comandos são executados a partir de `src/`:

### 1. Busca de Hiperparâmetros (Optuna)

```bash
python pipeline.py search -scenario <SCENARIO> [--nn CNN1D|MLP|LSTM] [--n_trials 30] [--timeout 3600]
```

### 2. Análise Pós-Trials

```bash
python pipeline.py post_trials -scenario <SCENARIO> [--nn CNN1D|MLP|LSTM]
```

### 3. Treinamento Final

```bash
python pipeline.py train -scenario <SCENARIO> [--nn CNN1D|MLP|LSTM] [--num_models 30] [--epochs 200]
```

## Pipeline de Análise (`analysis_pipeline.py`)

### SHAP — Importância de Features

```bash
python analysis_pipeline.py shap -scenario <SCENARIO> --nn <NN> [--background_size 100] [--sample_size 200]
```

### Curva de Aprendizado

```bash
python analysis_pipeline.py learning_curve -scenario <SCENARIO> --nn <NN> [--epochs 10]
```

### Agregar Métricas dos Modelos

```bash
python analysis_pipeline.py aggregate -scenario <SCENARIO> --nn <NN>
```

### Análise Global (todos os experimentos)

```bash
python analysis_pipeline.py analyze [--base_dir output] [--output_dir analise_global]
```

## Cenários Disponíveis

O nome do cenário segue o padrão `<posição>_T` (domínio temporal) ou `<posição>_F` (domínio de frequência).

| Cenário | Posição | Domínio | Shape de entrada |
|---|---|---|---|
| `chest_T` | chest | temporal | (1100, 8) |
| `chest_F` | chest | frequência | (550, 8) |
| `left_T` | left | temporal | (460, 8) |
| `left_F` | left | frequência | (230, 8) |
| `right_T` | right | temporal | (460, 8) |
| `right_F` | right | frequência | (230, 8) |
| `chest_left_T` | chest + left | temporal | (460, 16) |
| `chest_left_F` | chest + left | frequência | (230, 16) |
| `chest_right_T` | chest + right | temporal | (460, 16) |
| `chest_right_F` | chest + right | frequência | (230, 16) |
| `chest_left_right_T` | chest + left + right | temporal | (460, 24) |
| `chest_left_right_F` | chest + left + right | frequência | (230, 24) |

## Estrutura do Projeto

```
fall-detect/
├── requirements.txt
├── dataset/
│   ├── 0_raw/               # Dados brutos (ID1..ID15 / CHEST, LEFT, RIGHT)
│   ├── chest/data/ labels/  # Datasets gerados por posição
│   ├── left/
│   ├── right/
│   ├── chest_left/
│   ├── chest_right/
│   └── chest_left_right/
└── src/
    ├── config.py             # Configurações centralizadas
    ├── pipeline.py           # CLI: search | post_trials | train
    ├── analysis_pipeline.py  # CLI: shap | learning_curve | aggregate | analyze
    ├── utils.py              # Loop de treino, Optuna, métricas, visualizações
    ├── neural_networks.py    # Arquiteturas CNN1D, MLP, LSTM
    └── data/
        ├── generate_datasets.py       # Geração de datasets (sensores individuais)
        ├── generate_fused_dataset.py  # Geração de datasets fundidos
        ├── validate_dataset.py        # Validação de datasets individuais
        └── validate_fused_dataset.py  # Validação de datasets fundidos
```

## Saídas Geradas

As saídas são salvas em `output/<nn>/<scenario>/`:

### Busca de Hiperparâmetros
- `optuna_study.db` — Banco SQLite do Optuna
- `best_hyperparameters.json` — Melhores hiperparâmetros encontrados
- `test_data.npz` — Split treino/val e teste
- `optuna_trials.csv` — Histórico de todos os trials
- `param_importance.png/.html` — Importância dos hiperparâmetros

### Treinamento Final
- `model_X/model_X.pt` — Modelo salvo
- `model_X/metrics_model_X.csv` — Métricas por modelo
- `model_X/loss_curve_model_X.png` — Curva de loss
- `model_X/confusion_matrix_model_X.png` — Matriz de confusão
- `all_metrics.csv` / `summary_metrics.csv` — Métricas agregadas
- `metrics_boxplot.png` — Boxplot das métricas

### Análise SHAP
- `shap_values_*.npy` — Valores SHAP
- `shap_importance_*.csv/.png` — Importância por feature e por classe

### Curva de Aprendizado
- `learning_curve.csv` / `learning_curve.png`

### Análise Global (`analise_global/`)
- Boxplots, curvas ROC, matrizes de confusão, curvas de aprendizado e importância de features comparados entre todos os experimentos

## Exemplo de Fluxo Completo

```bash
cd src

# 1. Busca de hiperparâmetros
python pipeline.py search -scenario chest_T --nn CNN1D --n_trials 30

# 2. Relatório pós-trials
python pipeline.py post_trials -scenario chest_T --nn CNN1D

# 3. Treinamento final
python pipeline.py train -scenario chest_T --nn CNN1D --num_models 30

# 4. SHAP
python analysis_pipeline.py shap -scenario chest_T --nn CNN1D

# 5. Curva de aprendizado
python analysis_pipeline.py learning_curve -scenario chest_T --nn CNN1D

# 6. Agregar métricas
python analysis_pipeline.py aggregate -scenario chest_T --nn CNN1D

# 7. Análise global
python analysis_pipeline.py analyze
```