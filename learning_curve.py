import argparse
import os
import numpy as np
import torch
from config import Config
from utils import plot_learning_curve


def main():
    parser = argparse.ArgumentParser(description="Geração da curva de aprendizado (learning curve)")
    parser.add_argument("-scenario", required=True, choices=[
        "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
        "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
        "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
    ])
    parser.add_argument("-position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
    parser.add_argument("--nn", required=False, choices=["CNN1D", "MLP", "LSTM"])
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas para cada fração")
    args = parser.parse_args()

    position = args.position
    label_type = args.label_type
    scenario = args.scenario
    model_type_arg = args.nn
    num_labels = Config.get_num_labels(label_type)

    base_out = Config.get_output_dir(model_type_arg, position, scenario, label_type)

    # Carregar hiperparâmetros
    results = load_hyperparameters(base_out)
    best_params = results["best_params"]
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg

    # Carregar dados
    data_path = os.path.join(Config.DATA_PATH, position)
    label_path = os.path.join(Config.LABEL_PATH, position)
    scenario_config = Config.SCENARIOS[scenario]
    X = np.load(os.path.join(data_path, scenario_config[0]))
    y = np.load(os.path.join(label_path, Config.LABELS_DICT[label_type])).astype(np.int64)
    X_test, y_test = load_test_data(base_out)

    input_shape_dict = Config.get_input_shape_dict(scenario, position, model_type)
    if model_type == "CNN1D":
        input_shape = input_shape_dict["CNN1D"]
    elif model_type == "LSTM":
        input_shape = X.shape[1:]
    else:  # MLP
        input_shape = input_shape_dict["MLP"]

    plot_learning_curve(
        create_model_fn=lambda best_params, input_shape, num_labels: create_model(model_type, best_params, input_shape, num_labels),
        X_full=X, y_full=y,
        X_test=X_test, y_test=y_test,
        input_shape=input_shape,
        num_labels=num_labels,
        best_params=best_params,
        device=Config.DEVICE,
        output_dir=base_out,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main() 