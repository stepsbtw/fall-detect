models=(MLP)
scenarios=(left_T right_T chest_T left_right_T chest_left_T chest_right_T chest_left_right_T)

for model in "${models[@]}"; do

    python run.py --cross_sensor --model "$model" --scenario chest_T --scale
    python run.py --cross_sensor --model "$model" --scenario left_T --scale
    python run.py --cross_sensor --model "$model" --scenario right_T --scale

    #python run.py --multisensor --model "$model" --mode ensemble --scale
    #python run.py --multisensor --model "$model" --mode stacking --scale

    python run.py --fused_missing --model "$model" --scenario chest_left_T --test_scenario chest_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_T --test_scenario left_T --scale

    python run.py --fused_missing --model "$model" --scenario chest_right_T --test_scenario chest_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_right_T --test_scenario right_T --scale

    python run.py --fused_missing --model "$model" --scenario left_right_T --test_scenario left_T --scale
    python run.py --fused_missing --model "$model" --scenario left_right_T --test_scenario right_T --scale

    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario chest_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario left_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario right_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario chest_left_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario chest_right_T --scale
    python run.py --fused_missing --model "$model" --scenario chest_left_right_T --test_scenario left_right_T --scale

done
