import optuna
import torch
import os
from training import *
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_intermediate_values, plot_param_importances, plot_slice

from model_utils import build_model_from_trial, prepare_datasets
from optuna_utils import handle_best_model, export_optuna_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

positions = ["left", "right", "chest"]
scenarios = [
    "Sc1_acc_T", "Sc1_gyr_T", "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_3_T", "Sc_4_T",
    "Sc1_acc_F", "Sc1_gyr_F", "Sc_2_acc_F", "Sc_2_gyr_F", "Sc_3_F", "Sc_4_F"
]

label_type = "binary_one"

def run_optuna_for_combination(position, scenario):
    label_dir = f"./labels_and_data/labels/{position}"
    data_dir = f"./labels_and_data/data/{position}"
    save_dir = os.path.join("optuna_results", f"{scenario}_{position}")
    os.makedirs(save_dir, exist_ok=True)

    # Shared state
    best_model = None
    best_f1 = -1
    best_test_report = None
    best_model_filename = None
    best_train_loss = None
    best_valid_loss = None

    def objective(trial):
        nonlocal best_model, best_f1, best_test_report, best_model_filename, best_train_loss, best_valid_loss

        input_shape, num_labels, train_dl, val_dl, test_dl = prepare_datasets(
            position, label_type, scenario, label_dir, data_dir)

        model, model_type = build_model_from_trial(trial, input_shape, num_labels)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        y_train = train_dl.dataset.tensors[1]
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model, train_loss, valid_loss = fit(
            epochs=15, lr=lr, model=model,
            train_dl=train_dl, val_dl=val_dl,
            criterion=loss_fn, patience=3, trial=trial
        )

        _, dict_report, cm, _, _, auc = get_class_report(model, val_dl)
        f1 = dict_report.get("1", {}).get("f1-score", 0.0)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_model_filename, best_test_report = handle_best_model(
                model, model_type, trial, f1, input_shape, test_dl, save_dir
            )

        return f1

    print(f"\nRunning Optuna for: scenario={scenario}, position={position}")
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=30)

    for plot_func, filename in [
        (plot_intermediate_values, "intermediate_values.html"),
        (plot_param_importances, "param_importances.html"),
        (plot_slice, "param_slices.html"),
    ]:
        try:
            fig = plot_func(study)
            fig.write_html(os.path.join(save_dir, filename))
        except Exception as e:
            print(f"Failed to generate {filename}: {e}")

    print("Best hyperparameters:")
    print(study.best_trial.params)

    if best_model:
        torch.save(best_model, os.path.join(save_dir, best_model_filename))
        print(f"Best model saved: {best_model_filename}")

    export_optuna_results(study, save_dir, best_model_filename, best_test_report, best_train_loss, best_valid_loss)


if __name__ == "__main__":
    for position in positions:
        for scenario in scenarios:
            try:
                run_optuna_for_combination(position, scenario)
            except Exception as e:
                print(f"Error in scenario={scenario}, position={position}: {e}")
