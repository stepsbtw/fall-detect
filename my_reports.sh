#! /bin/bash

# Definindo as variáveis para cada loop
cenario=Sc_4_T
redeneural=("MLP" "CNN1D" "LSTM")
sensor="chest"
label="binary_two"

# Loop aninhado


for nn in "${redeneural[@]}"; do
    result_file="output/${nn}/${sensor}/${cenario}/${label}/optuna_trials.csv"

    if [ -f "$result_file" ]; then
        echo "Resultado já existe: $result_file — skip."
    else
        echo "Executando: ${cenario} ${nn} ${sensor} ${label}"
        python run.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
    fi
done
