MODELS=(MLP CNN1D LogisticRegression XGBoost)

for model in "${MODELS[@]}"; do
    for test_dataset in chest left right; do
        for train_dataset in chest left right; do
            [[ "$train_dataset" == "$test_dataset" ]] && continue
            python run.py --experiment cross_sensor --model "$model" --train_data "$train_dataset" --test_data "$test_dataset"
        done
    done

    for test_dataset in chest left right chest_left chest_right left_right; do
        python run.py --experiment missing_sensor --model "$model" --train_data chest_left_right --test_data "$test_dataset"
        python run.py --experiment missing_sensor --model "$model" --train_data chest_left_right --test_data "$test_dataset" --sensor_dropout
    done

    for test_dataset in chest left; do
        python run.py --experiment missing_sensor --model "$model" --train_data chest_left --test_data "$test_dataset"
        python run.py --experiment missing_sensor --model "$model" --train_data chest_left --test_data "$test_dataset" --sensor_dropout
    done

    for test_dataset in chest right; do
        python run.py --experiment missing_sensor --model "$model" --train_data chest_right --test_data "$test_dataset"
        python run.py --experiment missing_sensor --model "$model" --train_data chest_right --test_data "$test_dataset" --sensor_dropout
    done

    for test_dataset in left right; do
        python run.py --experiment missing_sensor --model "$model" --train_data left_right --test_data "$test_dataset"
        python run.py --experiment missing_sensor --model "$model" --train_data left_right --test_data "$test_dataset" --sensor_dropout
    done
done

