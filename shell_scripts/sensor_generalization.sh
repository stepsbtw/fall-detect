# training on one sensor and testing on others 

#models=(CNN1D MLP LSTM XGBoost RF CatBoost SVM GRU)
models=(CNN1D MLP LSTM XGBoost)
scenarios=(left_T right_T chest_T)

for model in "${models[@]}"; do
    for scenario in "${scenarios[@]}"; do
        python run.py --train --model $model --cross_sensor --train_scenario $scenario --scale
    done
done