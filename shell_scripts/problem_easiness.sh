# train a logistic regression model to predict the easiness of a problem

scenarios=(left_T chest_T right_T chest_left_T chest_right_T chest_left_right_T)

for scenario in "${scenarios[@]}"; do   
    python run.py --train --model LogisticRegression --scenario $scenario --scale
done