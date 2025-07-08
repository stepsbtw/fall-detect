# Fall Detect PyTorch

Adaptação para o PyTorch do trabalho original : https://AILAB-CEFET-RJ/falldetection

Baseado no artigo - A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, K-Fold Cross Validation e o modelo LSTM foi implementado e testado junto aos demais.

Os sensores no trabalho original não são combinados (left, chest, right), por motivos de "escolher o melhor".

Suporta 3 arquiteturas de redes neurais, **CNN1D**, **MLP** e **LSTM**, com **otimização bayesiana de hiperparâmetros via Optuna**.

---

## Estrutura do Projeto

```
project/
│
├── generate_datasets.py        # Ponto de partida: cria rótulos e datasets a partir do IPQM-Fall
├── run.py                      # Arquivo principal: CLI, carregamento, treinamento e salvamento
├── train.py                    # Lógica de treinamento, avaliação e Optuna
├── neural_networks.py          # Arquiteturas: CNN1D, MLP e LSTM
├── labels_and_data/            # Dados e rótulos organizados por posição do sensor
│   ├── data/
│   └── labels/
└── output/                     # Criado automaticamente: armazena modelos, métricas e gráficos
```

---

## Início Rápido

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

### Alternativa 1: Execute o Treinamento com Optuna via CLI

```bash
python run.py \
  -scenario Sc1_acc_T \
  -position chest \
  -label_type binary_one \
  --nn CNN1D
```

#### Argumentos:

| Argumento                | Opções                                               | Descrição                                      |
|--------------------------|------------------------------------------------------|------------------------------------------------|
| `-scenario`             | Sc1_acc_T, Sc1_gyr_T, ..., Sc_4_F                    | Tipo de sinal e domínio                        |
| `-position`             | left, chest, right                                   | Posição do sensor                              |
| `-label_type`           | multiple_one, multiple_two, binary_one, binary_two   | Tipo de rótulo (target)                        |
| `--nn` (opcional) | CNN1D, MLP, LSTM                        | Arquitetura da rede neural                     |

---

## Alternativa 2: Use o Script Automatizado

Treinará todas as possíveis combinações de modelos, cenários, rótulos e sensores, efetuará a busca de hiperparâmetros com o Optuna.

```bash
./extract_reports.sh
```

## Arquiteturas Suportadas

- **CNN1DNet**: Convoluções 1D para sinais no tempo ou frequência  
- **MLPNet**: Rede neural totalmente conectada para vetores planos  
- **LSTMNet**: Modelo recorrente para sequências temporais  

---

## Funcionalidades

- Otimização de hiperparâmetros com **Optuna**
- Treinamento e avaliação automáticos
- Suporte a classificação **binária** (e futuramente **multiclasse**)
- Salva:
  - Modelos `.pt`
  - Matrizes de confusão
  - Relatórios de classificação
  - Curvas ROC (para problemas binários)
  - Métricas como MCC, acurácia, sensibilidade, especificidade e precisão

---

## Exemplo de Saída

```
output/
└── lstm/
    └── chest/
        └── Sc1_acc_T/
            └── binary_one/
                ├── model_1/
                │   ├── model_1.pt
                │   ├── confusion_matrix_model_1.png
                │   ├── classification_report_model_1.txt
                │   ├── roc_curve_model_1.png
                │   └── metrics_model_1.csv
                └── model_2/
                    ...
```

---

## Observações

- A otimização via Optuna é repetida por **30 experimentos (trials)**
- O treinamento final é repetido **20 vezes** para garantir robustez ( ja feito no trabalho original )
- **Early Stopping** está implementado com patience = 5
- Utiliza **Optuna Median Pruner** para interromper execuções com baixo desempenho
- K-Fold por enquanto com 5 Folds (Datasets são pequenos, 1040*6(chest) e 408*6(left).
- Batch Sizes pra CNN e LSTM = 32, MLP = 64 (adaptável)
- Suporte a paralelismo com PyTorch se necessário.

---

## Requisitos

Para minimizar conflitos, priorize utilizar o Python 3.10
