cenarios=("Sc_4_T")
redeneural=("MLP" "CNN1D" "LSTM")
sensores=("chest")
labels=("binary_two")

for cenario in "${cenarios[@]}"; do
  for sensor in "${sensores[@]}"; do
    for label in "${labels[@]}"; do
      for nn in "${redeneural[@]}"; do
        outdir="output/${nn}/${sensor}/${cenario}/${label}"
        trials_file="${outdir}/optuna_trials.csv"
        best_file="${outdir}/best_hyperparameters.json"
        summary_file="${outdir}/summary_metrics.csv"

        # Busca de hiperparâmetros
        if [ ! -f "$trials_file" ]; then
          echo "Buscando hiperparâmetros: $cenario $nn $sensor $label"
          python post_trials.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
        else
          echo "Trials já existem: $trials_file"
        fi

        # Treinamento final ou SHAP direto
        if [ -f "$best_file" ]; then
          if [ -f "$summary_file" ]; then
            echo "summary_metrics.csv já existe — PULANDO treinamento final. Rodando SHAP."
            python shap_importance.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
          else
            echo "Treinando modelos finais: $cenario $nn $sensor $label"
            python final_training.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn" --num_models 20
          fi
        else
          echo "Hiperparâmetros não encontrados, pulando treinamento final e SHAP."
        fi

      done
    done
  done
done
