# training on 14 people and testing on the remaining one (leave-one-subject-out cross-validation)
# testing if cross entropy loss with class weights improves the results for the imbalanced classes (falls are less frequent than non-falls)
# testing if using only the magnitude (instead of the 3 axes) improves the results
# testing if using only the 3 axes (instead of the magnitude) improves the results
# testing if using bigger validation groups improves the loss curves and early stopping
# testing if scaling the data improves the results

models=(LogisticRegression)
scenarios=(left_T right_T chest_T left_right_T chest_left_T chest_right_T chest_left_right_T)

for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python run.py --train --model $model --scenario $scenario
        python run.py --train --model $model --scenario $scenario --scale
        python run.py --model $model --cross_sensor --scenario $scenario --scale
        python run.py --fused_missing --model $model --scenario $train_scenario --test_scenario $test_scenario --scale
        python run.py --multisensor --model $model --scenario $scenario --mode ensemble --scale
        python run.py --multisensor --model $model --scenario $scenario --mode stacking --scale
    done
done