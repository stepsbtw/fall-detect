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
            echo "summary_metrics.csv já existe — PULANDO treinamento final. Rodando Permutation Importance."
            python permutation_importance.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
            lc_metrics_file="${outdir}/learning_curve_metrics.csv"
            if [ ! -f "$lc_metrics_file" ]; then
              echo "Rodando curva de aprendizado (learning curve) para $cenario $nn $sensor $label"
              python learning_curve.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
            else
              echo "Learning curve já existe: $lc_metrics_file — pulando."
            fi
          else
            echo "Treinando modelos finais: $cenario $nn $sensor $label"
            python final_training.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn" --num_models 20
          fi
        else
          echo "Hiperparâmetros não encontrados, pulando treinamento final e Permutation Importance."
        fi

      done
    done
  done
done
