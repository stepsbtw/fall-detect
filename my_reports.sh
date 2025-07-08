#! /bin/bash

# Definindo as variáveis para cada loop
cenarios=(Sc_4_T)
redeneural=("MLP" "CNN1D" "LSTM")
sensores=("chest")
labels=("binary_two")

# Loop aninhado

for label in "${labels[@]}"; do
    for cenario in "${cenarios[@]}"; do
        for sensor in "${sensores[@]}"; do
            for nn in "${redeneural[@]}"; do
                result_file="output/optuna/${nn}/${sensor}/${cenario}/${label}/optuna_trials.csv"

                if [ -f "$result_file" ]; then
                    echo "Resultado já existe: $result_file — skip."
                else
                    echo "Executando: ${cenario} ${nn} ${sensor} ${label}"
                    python run.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
                fi
            done
        done
    done
done
