#! /bin/bash


# Definindo as variáveis para cada loop
cenarios=("Sc1_acc_T" "Sc1_gyr_T" "Sc1_acc_F" "Sc1_gyr_F" "Sc_2_acc_T" "Sc_2_gyr_T" "Sc_2_acc_F" "Sc_2_gyr_F" "Sc_3_T" "Sc_3_F" "Sc_4_T" "Sc_4_F")
redenerural=("MLP" "CNN1D")
sensores=("chest" "right" "left")
contador=0
# Loop aninhado
for cenario in "${cenarios[@]}"; do
    for nn in "${redenerural[@]}"; do
        for sensor in "${sensores[@]}"; do
		contador=$((contador + 1))
		printf "%s %s %s\n" "$cenario" "$nn" "$sensor"
		python3 training.py -s "$cenario" -p "$sensor" -nn "$nn"
        done
    done
done
