# Fall-Detect

### Antes de tudo
Com o objetivo de isolar o projeto, é recomendado a criação de um ambiente virtual do python:

```
python3 -m venv nome-do-enviroment
```

## Importação dos dados

Para execução desse repositório, será necessário utilizar os requisitos descritos no arquivo requirements.txt. É possivel instalar todos os pacotes necessários com o unico comando abaixo:

```
cd fall-detect/
pip install -r requirements.txt
```

Após instalação dos pacotes necessários, importaremos a base de dados a ser utilizada. Essa a base de dados encontra-se diposnível publicamente [aqui](https://zenodo.org/records/12760391). 

## Geração dos datasets 

Uma vez com os pacotes necessários instalados e a base de dados baixada e **descompactada**. Será realizado a criação do dataset através dos comandos:
```
python3 training_data_generator.py chest
python3 training_data_generator.py right
python3 training_data_generator.py left
```

Uma vez com o dataset de cada modalidade (chest, right e left) criado é possivel seguir para a etapa de treinamento da Rede Neural.

## Treinamento e Plotagem 
Para o treinamento, execute o script `training.py` com os parâmetros que deseja como **cenários**, **sensor**, **tipo de classificação**, etc. Em caso de dúvidas, verifique seção de **--help** do script.

```
------------------------------------------------------------------------------------------
Exemplos: 

python3 training.py -s Sc1_acc_T -p chest -l binary_one -nn CNN1D -c 2 -d 3
python3 training.py -s Sc_2_acc_F -p chest -nn CNN1D --n_conv 2 --n_dense 3 --epochs 10
------------------------------------------------------------------------------------------


> python3 training.py -s Sc_2_gyr_T -p chest -nn MLP --epochs 50

------------------------------------------------------------------------------------------
Datasets | Labels
------------------------------------------------------------------------------------------
Treinamento: torch.Size([3622, 3, 1020]) | torch.Size([3622]) | 3182(87%)-440(12%)
Validação: torch.Size([967, 3, 1020]) | torch.Size([967]) | 849(87%)-118(12%)
Teste: torch.Size([1449, 3, 1020]) | torch.Size([1449]) | 1273(87%)-176(12%)
------------------------------------------------------------------------------------------
Arquitetura da Rede Neural: 
...
...
...
------------------------------------------------------------------------------------------
[ 1/50] train_loss: 1.01968 valid_loss: 0.44025
[ 2/50] train_loss: 0.45207 valid_loss: 0.42807
[ 3/50] train_loss: 0.44221 valid_loss: 0.42522
...
[49/50] train_loss: 0.42600 valid_loss: 0.42295
[50/50] train_loss: 0.43025 valid_loss: 0.42292
Gráfico de Perda gerado com sucesso. (Verifique o diretório ...)
...

Relatório de classificação no dataset de treino:
              precision    recall  f1-score   support

           0       0.81      0.91      0.86       176
           1       0.99      0.97      0.98      1273

    accuracy                           0.96      1449
   macro avg       0.90      0.94      0.92      1449
weighted avg       0.97      0.96      0.96      1449

```

Após o treinamento será gerado um grafico, no diretório indicado, para análise do desempenho da rede neural ao longo do treinamento e será aplicado o `classification_report` com o dataset de teste.
---

#### Observações
Alguns arquivos presentes no repositório servem apenas como comparação com o projeto original (`run_of_the_neural_network_model.py` ou `model_builders/`) ou auxilio (`commands.txt`).