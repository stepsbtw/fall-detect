# testing if the model can generalize to armed and unarmed people.
set -euo pipefail

MODELS=(MLP CNN1D LogisticRegression XGBoost)
DATASETS=(chest left right)

for model in "${MODELS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		armed_dataset="${dataset}_armed"
		unarmed_dataset="${dataset}_unarmed"

		# Train in each condition independently
		python run.py --experiment train --model "$model" --train_data "$armed_dataset"
		python run.py --experiment train --model "$model" --train_data "$unarmed_dataset"

		# Cross-condition evaluation in both directions
		python run.py --experiment cross_sensor --model "$model" --train_data "$armed_dataset" --test_data "$unarmed_dataset"
		python run.py --experiment cross_sensor --model "$model" --train_data "$unarmed_dataset" --test_data "$armed_dataset"

        # Train on both conditions and test on each condition separately
        python run.py --experiment cross_sensor --model "$model" --train_data "$dataset" --test_data "$armed_dataset"
        python run.py --experiment cross_sensor --model "$model" --train_data "$dataset" --test_data "$unarmed_dataset"
	done
done