#!/bin/bash

# Fall Detection Pipeline - Script automatizado completo
# Executa busca de hiperparâmetros, treinamento final, análise de importância e SHAP

echo "=== Fall Detection Pipeline ==="
echo "Iniciando pipeline completo..."

# Configurações
cenarios=("Sc_4_T")
redeneural=("MLP" "CNN1D" "LSTM")
sensores=("chest")
labels=("binary_two")

# Validação inicial dos datasets
echo "Validando datasets..."
python validate_datasets.py

# --- ETAPA 1: Treinamento final e agregação de métricas ---
for cenario in "${cenarios[@]}"; do
  for sensor in "${sensores[@]}"; do
    for label in "${labels[@]}"; do
      for nn in "${redeneural[@]}"; do
        echo ""
        echo "=== Processando: $cenario $nn $sensor $label ==="
        
        outdir="output/${nn}/${sensor}/${cenario}/${label}"
        best_file="${outdir}/best_hyperparameters.json"
        summary_file="${outdir}/summary_metrics.csv"

        if [ -f "$best_file" ]; then
          if [ ! -f "$summary_file" ]; then
            echo "Treinando modelos finais: $cenario $nn $sensor $label"
            python final_training.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn" --num_models 20
            if [ $? -ne 0 ]; then
              echo "Erro no treinamento final. Pulando..."
              continue
            fi
            echo "Agregando métricas dos modelos finais..."
            python aggregate_metrics.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
            if [ $? -ne 0 ]; then
              echo "Erro na agregação de métricas. Pulando..."
              continue
            fi
          else
            echo "summary_metrics.csv já existe — PULANDO treinamento final."
          fi
        else
          echo "Hiperparâmetros não encontrados, pulando treinamento final e análises."
        fi
      done
    done
  done
fi

# --- ETAPA 2: Análises (após todos os modelos estarem prontos) ---
for cenario in "${cenarios[@]}"; do
  for sensor in "${sensores[@]}"; do
    for label in "${labels[@]}"; do
      for nn in "${redeneural[@]}"; do
        echo ""
        echo "=== Análises: $cenario $nn $sensor $label ==="
        outdir="output/${nn}/${sensor}/${cenario}/${label}"
        summary_file="${outdir}/summary_metrics.csv"
        shap_file="${outdir}/shap_analysis_completed.flag"
        lc_metrics_file="${outdir}/learning_curve_metrics.csv"
        if [ -f "$summary_file" ]; then
          # Permutation Importance
          echo "Rodando Permutation Importance..."
          python permutation_importance.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
          # Learning Curve
          if [ ! -f "$lc_metrics_file" ]; then
            echo "Rodando curva de aprendizado (learning curve) para $cenario $nn $sensor $label"
            python learning_curve.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
          else
            echo "Learning curve já existe: $lc_metrics_file — pulando."
          fi
          # SHAP Analysis
          if [ ! -f "$shap_file" ]; then
            echo "Rodando análise SHAP..."
            python shap_importance.py -scenario "$cenario" -position "$sensor" -label_type "$label" --nn "$nn"
            if [ $? -eq 0 ]; then
              touch "$shap_file"
              echo "Análise SHAP concluída."
            else
              echo "Erro na análise SHAP."
            fi
          else
            echo "Análise SHAP já foi executada."
          fi
        else
          echo "summary_metrics.csv não encontrado, pulando análises."
        fi
      done
    done
  done
fi

echo ""
echo "=== Análise Global ==="
echo "Gerando análise global dos resultados..."
python analysis.py

echo ""
echo "=== Pipeline Concluído ==="
echo "Todos os experimentos foram processados."
echo "Resultados disponíveis em:"
echo "   - output/: Resultados individuais"
echo "   - analise_global/: Análises agregadas"
echo "   - analise_global/shap/: Análises SHAP"
