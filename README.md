# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original: https://github.com/AILAB-CEFET-RJ/falldetection

Baseado no artigo (preprint) — A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, **Leave-One-Group-Out (LOGO) Cross Validation**, o modelo LSTM e a fusão de sensores foram implementados e testados junto aos demais.

Suporta **3 arquiteturas de redes neurais** (CNN1D, MLP, LSTM) e **3 modelos clássicos** (RF, SVM, XGBoost), com **otimização bayesiana de hiperparâmetros via Optuna** e suporte a parâmetros padrão sem busca prévia.

## Funcionalidades

- **3 Arquiteturas de Redes Neurais**: CNN1D, MLP, LSTM (PyTorch)
- **3 Modelos Clássicos**: Random Forest (RF), SVM, XGBoost (scikit-learn / xgboost)
- **Otimização Bayesiana**: Via Optuna com Early Stopping e Median Pruning; **F2-score (β=2)** como objetivo (ênfase em recall para detecção de quedas)
- **Parâmetros Padrão**: Treinamento sem busca Optuna prévia com `DEFAULT_PARAMS` por modelo
- **Tratamento de Desbalanceamento**: `CrossEntropyLoss` ponderada (redes neurais) e `class_weight='balanced'` / `scale_pos_weight` (modelos clássicos)
- **6 Posições de Sensor**: individuais (chest, left, right) e fusões (chest\_left, chest\_right, chest\_left\_right)
- **Domínio Temporal e de Frequência**: todos os cenários disponíveis nos dois domínios
- **Validação Cruzada LOGO**: Leave-One-Subject-Out por participante
- **Duas estratégias de avaliação**: holdout fixo de 3 sujeitos + LOGO aninhado (ver abaixo)
- **Split por Indivíduo**: Separação treino/val e teste feita a nível de participante (sem vazamento de dados)
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
- **Hiperparâmetros**: Ranges para otimização por arquitetura (`MODEL_CONFIGS`)
- **Parâmetros Padrão**: `DEFAULT_PARAMS` por modelo, usado quando não há busca Optuna
- **Treinamento**: epochs, patience, batch\_size, etc.
- **Split de dados**: 15 indivíduos; estratégia depende do modo (ver Pipeline abaixo)
- **Modelos clássicos**: `CLASSICAL_MODELS = {"RF", "SVM", "XGBoost"}`

## Estratégias de Avaliação

O pipeline oferece duas estratégias, ambas sem vazamento de dados entre sujeitos:

### Holdout Fixo (`search` + `train`)

```
15 sujeitos
├── 3 sujeitos de teste (fixos, bloqueados desde o início)
└── 12 sujeitos de treino/val
      ├── Optuna: LOGO 12-fold → seleciona HPs
      └── train: LOGO 12-fold (11 treino / 1 val para early stopping)
                 → avalia nos 3 sujeitos de teste fixos
                 → reporta média ± desvio sobre os 12 folds
```

**Prós:** 1 rodada Optuna; rápido.  
**Contra:** métrica final baseada em apenas 3 sujeitos.

### LOGO Aninhado (`nested`) — padrão ouro

```
15 sujeitos
└── LOGO externo (15-fold):
      para cada fold externo (sujeito i de teste):
        └── Optuna: LOGO interno sobre 14 sujeitos → HPs para este fold
            treina com esses HPs
            avalia no sujeito i
      → reporta média ± desvio sobre os 15 folds
```

**Prós:** zero leakage de HPs; todos os 15 sujeitos contribuem para o teste.  
**Contra:** 15× mais compute (15 rodadas Optuna).

## Pipeline de Treinamento (`pipeline.py`)

Todos os comandos são executados a partir de `src/`.

Os modelos disponíveis são: `CNN1D`, `MLP`, `LSTM` (redes neurais) e `RF`, `SVM`, `XGBoost` (modelos clássicos).

### 1. Busca de Hiperparâmetros (Optuna) — para estratégia holdout

```bash
python pipeline.py search -scenario <SCENARIO> --model <MODEL> [--n_trials 30] [--timeout 3600]
```

Roda LOGO sobre os **12 sujeitos de treino**. Salva `best_hyperparameters.json` e `test_data.npz` (com os 3 sujeitos de teste bloqueados).

### 2. Análise Pós-Trials

```bash
python pipeline.py post_trials -scenario <SCENARIO> --model <MODEL>
```

### 3. Avaliação LOGO com Holdout Fixo (`train`)

Requer `search` executado previamente.

```bash
python pipeline.py train -scenario <SCENARIO> --model <MODEL> [--num_models 1] [--epochs 200]
```

Cada fold treina em 11 sujeitos, usa o 12º como validação (early stopping), e avalia nos **3 sujeitos de teste fixos**.
`--num_models` controla o número de repetições LOGO completas (padrão=1 → 12 folds).

### 4. LOGO Aninhado (`nested`) — padrão ouro

Autossuficiente; não requer `search` prévio.

```bash
python pipeline.py nested -scenario <SCENARIO> --model <MODEL> [--n_trials 15] [--epochs 200]
```

Para cada um dos 15 sujeitos como fold externo, roda Optuna interno sobre os 14 restantes, treina com os melhores HPs e avalia no sujeito deixado de fora.

> **Nota:** para modelos clássicos (RF, SVM, XGBoost), `--epochs` é ignorado.

## Pipeline de Análise (`analysis_pipeline.py`)

### SHAP — Importância de Features

```bash
python analysis_pipeline.py shap -scenario <SCENARIO> --model <MODEL> [--background_size 100] [--sample_size 200]
```

### Curva de Aprendizado

```bash
python analysis_pipeline.py learning_curve -scenario <SCENARIO> --model <MODEL> [--epochs 10]
```

### Agregar Métricas dos Modelos

```bash
python analysis_pipeline.py aggregate -scenario <SCENARIO> --model <MODEL>
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
├── run.py                    # Script em lote: busca e/ou avaliação de todos os modelos/cenários
├── dataset/
│   ├── 0_raw/               # Dados brutos (ID1..ID15 / CHEST, LEFT, RIGHT)
│   ├── chest/data/ labels/  # Datasets gerados por posição
│   ├── left/
│   ├── right/
│   ├── chest_left/
│   ├── chest_right/
│   └── chest_left_right/
└── src/
    ├── config.py             # Configurações centralizadas (modelos, splits, DEFAULT_PARAMS)
    ├── pipeline.py           # CLI: search | post_trials | train | nested
    ├── analysis_pipeline.py  # CLI: shap | learning_curve | aggregate | analyze
    ├── utils.py              # Loop de treino, Optuna, métricas, visualizações, modelos clássicos
    ├── neural_networks.py    # Arquiteturas CNN1D (+ BatchNorm), MLP, LSTM
    └── data/
        ├── generate_datasets.py       # Geração de datasets (sensores individuais)
        ├── generate_fused_dataset.py  # Geração de datasets fundidos
        ├── validate_dataset.py        # Validação de datasets individuais
        └── validate_fused_dataset.py  # Validação de datasets fundidos
```

## Saídas Geradas

As saídas são salvas em `output/<model>/<scenario>/`:

### Holdout fixo (`train`)
- `rep_<N>/fold_s<subject>/` — resultados por fold
  - `losses_*.csv`, `loss_curve_*.png` — curvas de loss
  - `confusion_matrix_*.png`, `roc_curve_*.png` — métricas visuais
  - `metrics_*.csv` — métricas numéricas

### LOGO aninhado (`nested`)
- `nested/outer_s<subject>/` — um diretório por fold externo
  - `best_hyperparameters.json` — HPs selecionados pelo Optuna interno
  - Mesmos artefatos de métricas do modo `train`

### Busca de Hiperparâmetros
- `optuna_study.db` — Banco SQLite do Optuna
- `best_hyperparameters.json` — Melhores hiperparâmetros encontrados
- `test_data.npz` — Split treino/val e teste (por indivíduo)
- `optuna_trials.csv` — Histórico de todos os trials
- `param_importance.png/.html` — Importância dos hiperparâmetros

### Treinamento Final — Redes Neurais (CNN1D, MLP, LSTM)
- `model_X/model_X.pt` — Modelo salvo
- `model_X/metrics_model_X.csv` — Métricas por modelo
- `model_X/loss_curve_model_X.png` — Curva de loss
- `model_X/confusion_matrix_model_X.png` — Matriz de confusão
- `all_metrics.csv` / `summary_metrics.csv` — Métricas agregadas
- `metrics_boxplot.png` — Boxplot das métricas

### Treinamento Final — Modelos Clássicos (RF, SVM, XGBoost)
- `model_X/model_X.pkl` — Modelo serializado (joblib)
- `model_X/metrics_model_X.csv` — Métricas por modelo
- `model_X/confusion_matrix_model_X.png` — Matriz de confusão
- `model_X/roc_curve_model_X.png` — Curva ROC
- `model_X/classification_report_model_X.txt` — Relatório de classificação
- `all_metrics.csv` / `summary_metrics.csv` — Métricas agregadas

### Análise SHAP
- `shap_values_*.npy` — Valores SHAP
- `shap_importance_*.csv/.png` — Importância por feature e por classe

### Curva de Aprendizado
- `learning_curve.csv` / `learning_curve.png`

### Análise Global (`analise_global/`)
- Boxplots, curvas ROC, matrizes de confusão, curvas de aprendizado e importância de features comparados entre todos os experimentos

## Script em Lote (`run.py`)

O `run.py` executa busca e/ou treinamento para todos os modelos e cenários de uma vez.

```bash
# Treinar tudo com DEFAULT_PARAMS (sem busca)
python run.py

# Busca de hiperparâmetros + treinamento, todos os combos
python run.py --search --train

# Apenas busca, um modelo, todos os cenários
python run.py --search --model RF

# Busca + treino, um combo específico
python run.py --search --train --model CNN1D --scenario chest_T

# Configurações customizadas
python run.py --search --train --model LSTM --n_trials 50 --num_models 20 --epochs 300
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--search` | — | Executa busca Optuna |
| `--train` | — | Executa treinamento final (padrão se nenhuma flag for passada) |
| `--model` | todos | Filtra um modelo específico |
| `--scenario` | todos | Filtra um cenário específico |
| `--n_trials` | 30 | Número de trials Optuna |
| `--num_models` | 30 | Número de modelos treinados |
| `--epochs` | 200 | Épocas de treino (ignorado para RF/SVM/XGBoost) |

Logs são salvos em `logs/<model>_<scenario>_{search,train}.log`.

## Exemplo de Fluxo Completo

```bash
cd src

# --- Rede Neural (CNN1D) ---

# 1. Busca de hiperparâmetros
python pipeline.py search -scenario chest_T --model CNN1D --n_trials 30

# 2. Relatório pós-trials
python pipeline.py post_trials -scenario chest_T --model CNN1D

# 3. Treinamento final
python pipeline.py train -scenario chest_T --model CNN1D --num_models 30

# 4. SHAP
python analysis_pipeline.py shap -scenario chest_T --model CNN1D

# 5. Curva de aprendizado
python analysis_pipeline.py learning_curve -scenario chest_T --model CNN1D

# 6. Agregar métricas
python analysis_pipeline.py aggregate -scenario chest_T --model CNN1D

# --- Modelo Clássico (Random Forest) ---

# Com busca de hiperparâmetros
python pipeline.py search -scenario chest_T --model RF --n_trials 30
python pipeline.py train -scenario chest_T --model RF --num_models 10

# Sem busca prévia (usa DEFAULT_PARAMS)
python pipeline.py train -scenario chest_T --model RF --num_models 10

# --- Análise Global (todos os experimentos) ---

python analysis_pipeline.py analyze
```