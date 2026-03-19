# training on 14 people and testing on the remaining one (leave-one-subject-out cross-validation)

models=(CNN1D MLP XGBoost SVM) # LSTM RF CatBoost GRU)
scenarios=(chest_left_right_T)
#scenarios=(chest_left_T)

for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python run.py --train --model $model --scenario $scenario --scale
    done
done