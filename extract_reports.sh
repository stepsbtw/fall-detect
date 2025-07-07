#! /bin/bash

# Definindo as variáveis para cada loop
cenarios=(Sc_4_T Sc_4_F Sc_3_T Sc_3_F Sc_2_acc_T Sc_2_gyr_T Sc_2_acc_F Sc_2_gyr_F Sc1_acc_T Sc1_gyr_T Sc1_acc_F Sc1_gyr_F)
redenerural=("MLP" "CNN1D" "LSTM")
sensores=("chest" "right" "left")
labels=("binary_one" "binary_two")

# Loop aninhado
for cenario in "${cenarios[@]}"; do
    for nn in "${redenerural[@]}"; do
        for sensor in "${sensores[@]}"; do
            for label in "${labels[@]}"; do
                echo "Criando ${cenario}_${nn}_${sensor}_${label}.json..."
                python training.py -s "$cenario" -p "$sensor" -nn "$nn" -l "$label"
            done
        done
    done
done


# Agrupando resultados
python agg_results.py
