# training on one sensor and testing on others 

models=(LogisticRegression)
scenarios=(left_T right_T chest_T left_right_T chest_left_T chest_right_T chest_left_right_T)

for model in "${models[@]}"; do
    for scenario in "${scenarios[@]}"; do
        python run.py --model $model --cross_sensor --scenario $scenario --scale
        python run.py --fused_missing --model $model --scenario $train_scenario --test_scenario $test_scenario --scale
        python run.py --multisensor --model $model --scenario $scenario --mode ensemble --scale
        python run.py --multisensor --model $model --scenario $scenario --mode stacking --scale
    done
done