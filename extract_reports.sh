#!/bin/bash

RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"

cenarios=("Sc1_acc_T" "Sc1_gyr_T" "Sc1_acc_F" "Sc1_gyr_F" "Sc_2_acc_T" "Sc_2_gyr_T" "Sc_2_acc_F" "Sc_2_gyr_F" "Sc_3_T" "Sc_3_F" "Sc_4_T" "Sc_4_F")
redes=("MLP" "CNN1D")
sensores=("left" "chest" "right")
#taxas=("0.001" "0.0001")
#epocas=(10 20)

run_job() {
  cenario=$1
  nn=$2
  sensor=$3
  taxa=$4
  epoca=$5
  out="${RESULT_DIR}/${cenario}_${nn}_${sensor}.json"
  if [ ! -f "$out" ]; then
    echo "Rodando $out..."
    python training.py -s "$cenario" -p "$sensor" -nn "$nn" --export
  else
    echo "✅ Já existe: $out"
  fi
}

export -f run_job
export RESULT_DIR

# Gera as combinações e paraleliza
for c in "${cenarios[@]}"; do
  for r in "${redes[@]}"; do
    for s in "${sensores[@]}"; do
      echo "$c $r $s"
    done
  done
done | parallel -j 4 run_job {1} {2} {3}