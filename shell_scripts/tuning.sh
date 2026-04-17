models=(LogReg MLP CNN1D LSTM)
scenarios=(left_T right_T chest_T left_right_T chest_left_T chest_right_T)

for model in "${models[@]}"; do
    # python run.py --train --model "$model" --scenario chest_left_right_T --scale 
    python run.py --train --model "$model" --scenario chest_left_right_T --scale --tune_threshold
    python run.py --train --model "$model" --scenario chest_left_right_T --scale --sensor_dropout --sensor_dropout_p 0.5 --sensor_dropout_max_off 2 --tune_threshold
done

for model in "${models[@]}"; do
    for scenario in "${scenarios[@]}"; do
        # python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario "$scenario" --scale
        python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario "$scenario" --scale --tune_threshold
        python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario "$scenario" --scale --sensor_dropout --sensor_dropout_p 0.5 --sensor_dropout_max_off 2 --tune_threshold
    done
done