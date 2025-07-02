#! /bin/bash

# Definindo as variáveis para cada loop
cenarios=("Sc1_acc_T" "Sc1_gyr_T" "Sc1_acc_F" "Sc1_gyr_F" "Sc_2_acc_T" "Sc_2_gyr_T" "Sc_2_acc_F" "Sc_2_gyr_F" "Sc_3_T" "Sc_3_F" "Sc_4_T" "Sc_4_F")
redenerural=("MLP" "CNN1D")
sensores=("chest" "right" "left")

# Loop aninhado
for cenario in "${cenarios[@]}"; do
    for nn in "${redenerural[@]}"; do
        for sensor in "${sensores[@]}"; do
		if [ ! -e "./results/${cenario}_${nn}_${sensor}.json" ]; then
			echo "Criando ${cenario}_${nn}_${sensor}.json..."
			python training.py -s "$cenario" -p "$sensor" -nn "$nn" --export
		fi
        done
    done
done


# Agrupando resultados
python agg_results.py

# Verificando se houve alterações
if [[ `git status --porcelain output/ results/ models/` ]]; then
    echo "Mudanças detectadas. Commitando e enviando..."
    
    git add output/ results/ models/
    
    COMMIT_MSG="Resultado dos testes automatizados commit - $(date '+%Y-%m-%d %H:%M:%S')"
    git commit -m "$COMMIT_MSG"

    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git push origin "$CURRENT_BRANCH"

    echo "Alterações commitadas e enviadas com sucesso."
else
    echo "Nenhuma alteração detectada nas pastas output/, results/ ou models/. Nada foi commitado."
fi