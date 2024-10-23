# Escrito por Leandro Soares - https://github.com/SoaresLMB
import argparse
import os
from builders.model_builders import generate_training_testing_and_validation_sets, create_study_object, objective, \
    cnn1d_architecture, save_results, mlp_architecture, save_best_trial_to_csv

""" EXECUTION OF THE BAYESIAN OPTIMIZATION PROGRAM AND SUBSEQUENT MODEL TRAINING """
# ------------------------------------------------------------------------------------------


def parse_input():
    parser = argparse.ArgumentParser(
        description="Script for Bayesian optimization and model training")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
            "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
            "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
        ],
        required=True,
        help="Neural network scenario (e.g. Sc1_acc_F, Sc1_gyr_T, etc.)",
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["left", "chest", "right"],
        required=True,
        help="Sensor position (left, chest, right)",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["multiple_one", "multiple_two",
                 "binary_one", "binary_two"],
        required=True,
        help="Label type (multiple_one, multiple_two, binary_one, binary_two)",
    )
    parser.add_argument(
        "--neural_network_type",
        type=str,
        choices=["CNN1D", "MLP"],
        required=True,
        help="Tipo de rede neural (CNN1D ou MLP)",
    )
    args = parser.parse_args()

    return args.position, args.label_type, args.scenario, args.neural_network_type


label_filename = {
    "multiple_one": "multiple_class_label_1.npy",
    "binary_one": "binary_class_label_1.npy",
    "multiple_two": "multiple_class_label_2.npy",
    "binary_two": "binary_class_label_2.npy"
}

labels = {"multiple_one": 37, "multiple_two": 26,
          "binary_one": 2, "binary_two": 2
          }

array_sizes = {"chest": 1020, "right": 450, "left": 450}

current_directory = os.path.dirname(__file__)

# ------------------------------------------------------------------------------------------
position, label_type, scenario, neural_network_type = parse_input()
# ------------------------------------------------------------------------------------------

num_labels, array_size = labels.get(label_type), array_sizes.get(position)

# Diretório de dados
data_path = os.path.join(current_directory, "labels_and_data", "data", position)

# Diretório de rotulos - Targets?
label_path = os.path.join(current_directory, "labels_and_data", "labels", position)

# Muito confuso isso...
neural_network_scenarios = {
    # for Sc1_CNN1D_acc_T and Sc1_MLP_acc_T
    "Sc1_acc_T": [os.path.join(data_path, "magacc_time_domain_data_array.npy"), (array_size, 1)],
    # for Sc1_CNN1D_gyr_T and Sc1_MLP_gyr_T
    "Sc1_gyr_T": [os.path.join(data_path, "maggyr_time_domain_data_array.npy"), (array_size, 1)],
    # for Sc1_CNN1D_acc_F and Sc1_MLP_acc_F
    "Sc1_acc_F": [os.path.join(data_path, "magacc_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
    # for Sc1_CNN1D_gyr_F and Sc1_MLP_gyr_F
    "Sc1_gyr_F": [os.path.join(data_path, "maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 1)],

    # for Sc_2_CNN1D_acc_T and Sc_2_MLP_acc_T
    "Sc_2_acc_T": [os.path.join(data_path, "acc_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
    # for Sc_2_CNN1D_gyr_T and Sc_2_MLP_gyr_T
    "Sc_2_gyr_T": [os.path.join(data_path, "gyr_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
    # for Sc_2_CNN1D_acc_F and Sc_2_MLP_acc_F
    "Sc_2_acc_F": [os.path.join(data_path, "acc_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
    # for Sc_2_CNN1D_gyr_F and Sc_2_MLP_gyr_F
    "Sc_2_gyr_F": [os.path.join(data_path, "gyr_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],

    # for Sc_3_CNN1D_T and Sc_3_MLP_T
    "Sc_3_T": [os.path.join(data_path, "magacc_and_maggyr_time_domain_data_array.npy"), (array_size, 2)],
    # for Sc_3_CNN1D_F and Sc_3_MLP_F
    "Sc_3_F": [os.path.join(data_path, "magacc_and_maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 2)],

    # for Sc_4_CNN1D_T and Sc_4_MLP_T
    "Sc_4_T": [os.path.join(data_path, "acc_and_gyr_three_axes_time_domain_data_array.npy"), (array_size, 6)],
    # for Sc_4_CNN1D_F and Sc_4_MLP_F
    "Sc_4_F": [os.path.join(data_path, "acc_and_gyr_three_axes_frequency_domain_data_array.npy"), (int(array_size/2), 6)],
}

data = neural_network_scenarios[scenario]

input_shape = data[1] if neural_network_type == "CNN1D" else array_size

labels = os.path.join(label_path, label_filename.get(label_type))

#
X_train, X_test, y_train, y_test, X_val, y_val = generate_training_testing_and_validation_sets(data[0], labels)



output_dir = os.path.join(current_directory, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

neural_network_results_dir = os.path.join(
    output_dir, neural_network_type, f"{position}") if neural_network_type == "CNN1D" else os.path.join(output_dir, neural_network_type)

if not os.path.exists(neural_network_results_dir):
    os.makedirs(neural_network_results_dir)

scenario_dir = os.path.join(neural_network_results_dir, scenario, label_type)
if not os.path.exists(scenario_dir):
    os.makedirs(scenario_dir)

print("\nStarting Optimization\n")

# Estudar
best_trial, best_params = create_study_object(
    objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, scenario_dir, num_labels)

csv_file_path = os.path.join(scenario_dir, "best_trial.csv")
save_best_trial_to_csv(best_trial, best_params, csv_file_path)

print("\nStarting Training\n")

# Porque 21??
for i in range(1, 21):

    print("\n")
    print(f"Training Model {i}")
    print("\n")

    if neural_network_type == "CNN1D":

        mlp_output_dir = os.path.join(scenario_dir, f"CNN1D_model_{i}")
        if not os.path.exists(mlp_output_dir):
            os.makedirs(mlp_output_dir)

        model, historic = cnn1d_architecture(
            input_shape,
            X_train,
            y_train,
            X_val,
            y_val,
            best_params["filter_size"],
            best_params["kernel_size"],
            best_params["num_layers"],
            best_params["num_dense_layers"],
            best_params["dense_neurons"],
            best_params["dropout"],
            best_params["learning_rate"],
            num_labels)

        decision_threshold = best_params["decision_threshold"]

        save_results(model, historic, X_test, y_test, num_labels,
                     i, decision_threshold, mlp_output_dir, neural_network_type)

    elif neural_network_type == "MLP":

        mlp_output_dir = os.path.join(scenario_dir, f"MLP_model_{i}")
        if not os.path.exists(mlp_output_dir):
            os.makedirs(mlp_output_dir)

        model, historic = mlp_architecture(
            input_shape,
            X_train,
            y_train,
            X_val,
            y_val,
            best_params["num_layers"],
            best_params["dense_neurons"],
            best_params["dropout"],
            best_params["learning_rate"],
            num_labels)

        decision_threshold = best_params["decision_threshold"]
        save_results(model, historic, X_test, y_test, num_labels,
                     i, decision_threshold, mlp_output_dir, neural_network_type)
