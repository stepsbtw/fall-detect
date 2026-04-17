MODELS=(MLP CNN1D LogisticRegression XGBoost)
DATASETS=(chest left right chest_left_right)
FUSED_DATASETS=(chest_left chest_right left_right chest_left_right)
ABLATIONS=(acc gyr acc_gyr acc_magacc gyr_maggyr magacc maggyr magacc_maggyr acc_gyr_magacc_maggyr)

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            python run.py --experiment train --model "$model" --train_data "$dataset" --ablation "$ablation"
        done
    done

    for dataset in "${FUSED_DATASETS[@]}"; do
        python run.py --experiment train --model "$model" --train_data "$dataset" --sensor_dropout
        python run.py --experiment stacking --model "$model" --train_data "$dataset"
        python run.py --experiment bagging --model "$model" --test_data "$dataset"
    done

    python run.py --experiment stacking --model "$model" --train_data "chest_left_right"
done