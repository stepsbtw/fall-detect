import argparse
import os
import sys

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

BASE_DATASETS = (
    "chest",
    "left",
    "right",
    "chest_left",
    "chest_right",
    "left_right",
    "chest_left_right",
)
DATASET_ROOT = os.path.join(ROOT_DIR, "dataset")


def _discover_datasets():
    names = set(BASE_DATASETS)
    if os.path.isdir(DATASET_ROOT):
        for entry in os.listdir(DATASET_ROOT):
            candidate = os.path.join(DATASET_ROOT, entry)
            if not os.path.isdir(candidate):
                continue
            time_path = os.path.join(candidate, "data", "data_time_domain.npy")
            labels_path = os.path.join(candidate, "labels", "labels.npy")
            if os.path.isfile(time_path) and os.path.isfile(labels_path):
                names.add(entry)
    return tuple(sorted(names))


DATASETS = _discover_datasets()
OUTPUT_ROOT = os.path.join(ROOT_DIR, "output")


def _run_name_for_dataset(dataset_name, args):
    from src.config import DEFAULT_ABLATION
    from src.utils import build_run_name

    return build_run_name(
        dataset_name,
        sensor_dropout=getattr(args, "sensor_dropout", False),
        ablation=getattr(args, "ablation", DEFAULT_ABLATION),
    )

def _print_run_header(args, output_dir):
    print("=" * 90)
    print("[RUN] dispatch")
    print(f"  - experiment: {args.experiment}")
    print(f"  - model: {args.model}")
    if args.train_data:
        print(f"  - train_data: {args.train_data}")
    if args.test_data:
        print(f"  - test_data: {args.test_data}")
    print(f"  - sensor_dropout: {bool(args.sensor_dropout)}")
    print(f"  - output_dir: {output_dir}")
    print("=" * 90)

def _output_dir_for_args(args):
    if args.experiment == "train":
        run_name = _run_name_for_dataset(args.train_data, args)
    elif args.experiment == "bagging":
        run_name = f"bagging_{_run_name_for_dataset(args.test_data, args)}"
    elif args.experiment == "stacking":
        run_name = f"stacking_{_run_name_for_dataset(args.train_data, args)}"
    elif args.experiment == "cross_sensor":
        run_name = f"cross_sensor_{_run_name_for_dataset(args.train_data, args)}_to_{args.test_data}"
    elif args.experiment == "missing_sensor":
        train_run_name = _run_name_for_dataset(args.train_data, args)
        run_name = f"missing_sensor_{train_run_name}_on_{args.test_data}"

    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    return os.path.join(OUTPUT_ROOT, args.model, run_name)

def _run_is_complete(args):
    output_dir = _output_dir_for_args(args)
    done_flag = os.path.join(output_dir, "DONE")
    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    return output_dir, (os.path.exists(done_flag) or os.path.exists(summary_csv))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["train", "cross_sensor", "missing_sensor", "bagging", "stacking"], required=True)
    parser.add_argument("--model", choices=["MLP", "CNN1D", "LSTM", "GRU", "LinearModel", "LogisticRegression", "SVM", "DecisionTree", "XGBoost", "LightGBM", "RandomForest"], required=True)

    parser.add_argument("--train_data", choices=DATASETS)
    parser.add_argument("--test_data", choices=DATASETS)

    parser.add_argument("--sensor_dropout", action="store_true")

    parser.add_argument("--ablation", choices=["acc", "gyr", "acc_gyr", "acc_gyr_magacc_maggyr", "magacc_maggyr", "acc_magacc", "gyr_maggyr", "magacc", "maggyr"], default="acc_gyr_magacc_maggyr")

    args = parser.parse_args()

    output_dir, is_complete = _run_is_complete(args)
    _print_run_header(args, output_dir)
    if is_complete:
        print(f"Run already complete at: {output_dir} - skipping.")
        return

    if args.experiment == "train":
        if args.train_data is None: raise ValueError("--train_data is required for train")
        from train import train_experiment

        train_experiment(args)

    elif args.experiment == "cross_sensor":
        if args.test_data is None: raise ValueError("--test_data is required for cross_sensor")
        from eval import eval_cross_sensor_experiment

        eval_cross_sensor_experiment(args)

    elif args.experiment == "missing_sensor":
        if args.test_data is None: raise ValueError("--test_data is required for missing_sensor")
        from eval import eval_missing_sensor_experiment

        eval_missing_sensor_experiment(args)

    elif args.experiment == "bagging":
        if args.test_data is None: raise ValueError("--test_data is required for bagging")
        from eval import eval_bagging_experiment

        eval_bagging_experiment(args)

    elif args.experiment == "stacking":
        if args.train_data is None: raise ValueError("--train_data is required for stacking")
        from train import train_stacking_experiment

        train_stacking_experiment(args)


if __name__ == "__main__":
    main()
