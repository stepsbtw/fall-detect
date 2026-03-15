# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original: https://github.com/AILAB-CEFET-RJ/falldetection

Baseado no artigo (preprint) — A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, **Leave-One-Group-Out (LOGO) Cross Validation**, o modelo LSTM, a fusão de sensores e um LOGO aninhado com busca de hiperparâmetros por fold foram implementados.

Suporta **3 arquiteturas de redes neurais** (CNN1D, MLP, LSTM) e **4 modelos clássicos** (RF, SVM, XGBoost, CatBoost), com **otimização bayesiana de hiperparâmetros via Optuna** e suporte a parâmetros padrão sem busca prévia.

## Funcionalidades

- **3 Arquiteturas de Redes Neurais**: CNN1D, MLP, LSTM (PyTorch)
- **4 Modelos Clássicos**: Random Forest (RF), SVM (LinearSVC), XGBoost, CatBoost
- **Otimização Bayesiana**: Via Optuna com Early Stopping e Median Pruning; **F1-score (fall class)** como objetivo
- **Parâmetros Padrão**: Treinamento sem busca Optuna prévia com `DEFAULT_PARAMS` por modelo (`train` mode)
- **Tratamento de Desbalanceamento**: `CrossEntropyLoss` ponderada (redes neurais), `class_weight='balanced'` (RF/SVM) e `scale_pos_weight` (XGBoost/CatBoost)
- **SVM Escalável**: `LinearSVC` + `CalibratedClassifierCV` — O(n×d) vs O(n²×d) do kernel RBF, adequado para entradas de alta dimensão
- **5 Cenários ativos**: chest\_T, left\_T, right\_T, chest\_left\_T, chest\_right\_T
- **Domínio Temporal e de Frequência**: todos os cenários disponíveis nos dois domínios
- **Validação Cruzada LOGO**: Leave-One-Subject-Out sobre todos os 15 participantes
- **Duas estratégias de avaliação**: `train` (DEFAULT\_PARAMS, zero leakage) e `nested` (Optuna por fold, zero leakage de HPs)
- **Zero Data Leakage**: no modo `train`, o sujeito de teste jamais influencia treino ou early stopping; no modo `nested`, busca de HPs ocorre somente nos N-1 sujeitos internos
- **Análise Global**: Comparações entre modelos e cenários com boxplots, correlações e sumários

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
- **Hiperparâmetros**: Ranges para otimização por arquitetura (`MODEL_CONFIGS`)
- **Parâmetros Padrão**: `DEFAULT_PARAMS` por modelo, usado no modo `train`
- **Treinamento**: epochs, patience, batch\_size, etc.
- **Modelos clássicos**: `CLASSICAL_MODELS = {"RF", "SVM", "XGBoost", "CatBoost"}`

## Estratégias de Avaliação

O pipeline oferece duas estratégias, ambas com outer LOGO sobre todos os 15 sujeitos:

### `train` — LOGO com DEFAULT\_PARAMS (sem busca)

```
15 sujeitos
└── LOGO externo (15 folds):
      para cada fold (sujeito i de teste):
        • treina nos N-2 sujeitos restantes
        • early stopping num sujeito de validação rotativo (dos N-1; nunca o de teste)
        • avalia no sujeito i (jamais visto durante treino ou stopping)
      → reporta média ± desvio sobre os 15 folds
```

**Prós:** rápido, sem busca Optuna; zero leakage de dados e de HPs.

### `nested` — LOGO aninhado com Optuna interno (padrão ouro)

```
15 sujeitos
└── LOGO externo (15 folds):
      para cada fold (sujeito i de teste):
        └── Optuna interno (GroupKFold k=3 ou holdout) sobre N-1 sujeitos → HPs deste fold
            • treina nos N-1 sujeitos (1 retido pelo Optuna para early stopping)
            • avalia no sujeito i
      → reporta média ± desvio sobre os 15 folds
```

**Prós:** zero leakage de HPs; cada fold tem HPs próprios.
**Contra:** 15× mais compute (15 rodadas Optuna).

## Pipeline de Treinamento (`pipeline.py`)

Todos os comandos são executados a partir de `src/`.

Os modelos disponíveis são: `CNN1D`, `MLP`, `LSTM` (redes neurais) e `RF`, `SVM`, `XGBoost`, `CatBoost` (modelos clássicos).

### `train` — LOGO com parâmetros padrão

```bash
python pipeline.py train -scenario <SCENARIO> --model <MODEL> [--epochs 200]
```

Executa LOGO completo (15 folds) usando `Config.DEFAULT_PARAMS`. `--epochs` é ignorado para modelos clássicos.

### `nested` — LOGO aninhado com Optuna interno

```bash
python pipeline.py nested -scenario <SCENARIO> --model <MODEL> \
    [--n_trials 15] [--epochs 200] [--inner {kfold,holdout,none}]
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--n_trials` | 15 | Trials Optuna por fold externo |
| `--epochs` | 200 | Épocas de treino (ignorado para modelos clássicos) |
| `--inner` | `kfold` | Estratégia interna: `kfold`=GroupKFold(k=3), `holdout`=GroupShuffleSplit(n=1), `none`=in-sample |

## Pipeline de Análise (`analysis_pipeline.py`)

Todos os comandos são executados a partir de `src/`.

### SHAP — Importância de Features

```bash
python analysis_pipeline.py shap -scenario <SCENARIO> --model <MODEL> \
    [--background_size 100] [--sample_size 200]
```

### Curva de Aprendizado

```bash
python analysis_pipeline.py learning_curve -scenario <SCENARIO> --model <MODEL> [--epochs 10]
```

### Agregar Métricas dos Folds

```bash
python analysis_pipeline.py aggregate -scenario <SCENARIO> --model <MODEL>
```

Lê os CSVs de `fold_s*/metrics_model_s*.csv`, gera `all_metrics.csv`, `summary_metrics.csv` e visualizações.

### Análise Global (todos os experimentos)

```bash
python analysis_pipeline.py analyze [--base_dir output] [--output_dir analysis]
```

## Cenários Disponíveis

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

Cenários ativos por padrão em `run.py`: `chest_T`, `left_T`, `right_T`, `chest_left_T`, `chest_right_T`.

## Estrutura do Projeto

```
fall-detect/
├── requirements.txt
├── run.py                    # Script em lote: treino e/ou análise de todos os combos
├── dataset/
│   ├── 0_raw/               # Dados brutos (ID1..ID15 / CHEST, LEFT, RIGHT)
│   ├── chest/data/ labels/  # Datasets gerados por posição
│   ├── left/
│   ├── right/
│   ├── chest_left/
│   ├── chest_right/
│   └── chest_left_right/
├── logs/                     # Logs de execução por combo
├── output/                   # Resultados organizados por modelo/cenário
└── src/
    ├── config.py             # Configurações centralizadas
    ├── pipeline.py           # CLI: train | nested
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

As saídas são salvas em `output/<MODEL>/<SCENARIO>/`.

### Modo `train` — por fold
```
output/<MODEL>/<SCENARIO>/
└── fold_s<N>/
    ├── model_s<N>.pt / model_s<N>.pkl   # Modelo salvo
    ├── metrics_model_s<N>.csv           # Métricas do fold
    ├── losses_s<N>.csv                  # Loss por época
    ├── loss_curve_s<N>.png
    ├── confusion_matrix_model_s<N>.png
    ├── roc_curve_model_s<N>.png
    └── classification_report_model_s<N>.txt
```

### Modo `nested` — por fold externo
```
output/<MODEL>/<SCENARIO>/nested/
└── outer_s<N>/
    ├── best_hyperparameters.json   # HPs selecionados pelo Optuna interno
    ├── optuna_trials.csv
    ├── param_importance.png
    ├── optuna_study.db
    └── (mesmos artefatos de métricas do modo train)
```

### Após `aggregate`
```
output/<MODEL>/<SCENARIO>/
├── all_metrics.csv          # Métricas de todos os folds concatenadas
├── summary_metrics.csv      # Média e desvio padrão
├── metrics_boxplot.png
├── f1_histogram.png
├── f1_vs_accuracy.png
└── correlation_heatmap.png
```

### Após `analyze`
```
analysis/
├── summary_final_models.csv
└── boxplots/
    ├── all/    f1/    acc/    mcc/    prec/    sens/    spec/
    └── (boxplots comparativos entre todos os combos)
```

## Script em Lote (`run.py`)

O `run.py` executa treino e/ou análise para todos os modelos e cenários de uma vez.

```bash
# Treinar tudo com DEFAULT_PARAMS (padrão se nenhuma flag for passada)
python run.py

# LOGO aninhado, todos os combos
python run.py --nested

# Treinar um combo específico
python run.py --train --model CNN1D --scenario chest_T

# LOGO aninhado, modelo específico, 15 trials internos
python run.py --nested --model CNN1D --n_trials 15

# LOGO aninhado com holdout interno
python run.py --nested --inner holdout

# Treinar tudo e depois agregar + analisar
python run.py --train --analyze

# Apenas agregar + analisar (sem re-treinar)
python run.py --analyze

# Agregar + analisar apenas para um modelo
python run.py --analyze --model CNN1D
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--train` | — | LOGO com DEFAULT\_PARAMS (padrão se nenhuma flag) |
| `--nested` | — | LOGO aninhado com Optuna interno por fold |
| `--analyze` | — | Agrega métricas por fold e roda análise global |
| `--model` | todos | Filtra um modelo específico |
| `--scenario` | todos | Filtra um cenário específico |
| `--n_trials` | 30 | Trials Optuna por fold (modo `nested`) |
| `--epochs` | 200 | Épocas de treino (ignorado para modelos clássicos) |
| `--inner` | `kfold` | CV interno do `nested`: `kfold`, `holdout` ou `none` |

Logs são salvos em `logs/<MODEL>_<SCENARIO>_{train,nested,aggregate}.log`.

## Exemplo de Fluxo Completo

```bash
# A partir da raiz do projeto

# 1. Treinar todos os modelos e cenários com DEFAULT_PARAMS
python run.py --train

# 2. Agregar métricas e gerar análise global
python run.py --analyze

# --- Ou tudo de uma vez ---
python run.py --train --analyze

# --- LOGO aninhado (padrão ouro) para um combo ---
cd src
python pipeline.py nested -scenario chest_T --model CNN1D --n_trials 15 --epochs 200

# --- Análise individual ---
python analysis_pipeline.py aggregate -scenario chest_T --model CNN1D
python analysis_pipeline.py shap -scenario chest_T --model CNN1D
python analysis_pipeline.py learning_curve -scenario chest_T --model CNN1D
python analysis_pipeline.py analyze
```


