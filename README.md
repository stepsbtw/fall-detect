# Fall-Detect
Adaptação para o PyTorch do trabalho original : https://AILAB-CEFET-RJ/falldetection

Baseado no artigo - A Machine Learning Approach to Automatic Fall Detection of Soldiers: https://arxiv.org/abs/2501.15655v2

Além da adaptação, o modelo LSTM foi implementado e testado junto aos demais. 

Os sensores no trabalho original não são combinados (left, chest, right), por motivos de "escolher o melhor". 

Podemos propor uma fusão dos dados, assim podemos aproveitar as nuances de cada sensor e obter o melhor modelo.

### Antes de tudo
Com o objetivo de isolar o projeto, é recomendado a criação de um ambiente virtual do python:

```
python -m venv nome-do-enviroment
```

obs: Se possível, utilize o Python 3.10 para nenhum erro de compatibilidade entre bibliotecas do projeto.

## Importação dos dados

Para execução desse repositório, será necessário utilizar os requisitos descritos no arquivo requirements.txt. É possivel instalar todos os pacotes necessários com o unico comando abaixo:

```
cd fall-detect/
pip install -r requirements.txt
```

Após instalação dos pacotes necessários, importaremos a base de dados a ser utilizada. Essa a base de dados encontra-se disponível publicamente [aqui](https://zenodo.org/records/12760391). 

## Geração dos datasets 

Uma vez com os pacotes necessários instalados e a base de dados baixada e **descompactada**. Será realizado a criação do dataset através dos comandos:
```
python generate_datasets.py chest
python generate_datasets.py right
python generate_datasets.py left
```

Uma vez com o dataset de cada modalidade (chest, right e left) criado é possivel seguir para a etapa de treinamento da Rede Neural.

## Treinamento e Nested Cross Validation
Para o treinamento, execute o script `training_optuna.py`. O Optuna foi utilizado junto com Pruners para automatizar e otimizar o processo.
