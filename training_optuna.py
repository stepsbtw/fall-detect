import optuna
import os
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_intermediate_values, plot_param_importances, plot_slice
from utils_data import prepare_datasets
from utils_training import set_seed, build_model_optuna, fit
from utils_output import export_optuna_results, save_report, best_model_optuna

from optuna_analysis import analyze_logged_trials, log_optuna_trial, analyze_optuna_study
import gc

from sklearn.metrics import f1_score
import json


positions = ["left", "right", "chest"]
scenarios = [
    "Sc1_acc_T", "Sc1_gyr_T", "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_3_T", "Sc_4_T",
    "Sc1_acc_F", "Sc1_gyr_F", "Sc_2_acc_F", "Sc_2_gyr_F", "Sc_3_F", "Sc_4_F"
]

label_type = "binary_one"

def run_optuna_for_combination(position, scenario):
    study_name = f"{scenario}_{position}"
    save_dir = os.path.join("optuna_results", study_name)
    log_dir = os.path.join("optuna_logs", study_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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
            position, label_type, scenario,
            label_dir=f"./labels_and_data/labels/{position}",
            data_dir=f"./labels_and_data/data/{position}"
        )

        model_type = trial.suggest_categorical("model_type", ["MLP", "CNN1D", "LSTM"])
        model = build_model_optuna(model_type, trial, input_shape, num_labels)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device, non_blocking=True)

        y_train = train_dl.dataset.tensors[1]
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model, train_loss, valid_loss = fit(
            epochs=15, lr=lr, model=model,
            train_dl=train_dl, val_dl=val_dl,
            criterion=loss_fn, patience=3, trial=trial
        )

        eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        report, dict_report, conf_matrix, all_labels, all_probs, all_predictions, auc = save_report(eval_model, val_dl)
        f1 = f1_score(all_labels, all_predictions, pos_label=1)

        print(f"[Trial {trial.number}] F1: {f1:.4f}, model_type={model_type}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_model_filename, best_test_report = best_model_optuna(
                model, model_type, trial, f1, input_shape, test_dl, save_dir
            )

        log_optuna_trial(trial, f1, save_dir=log_dir)
        torch.cuda.empty_cache()
        gc.collect()
        return f1

    print(f"\n Running Optuna for: scenario={scenario}, position={position}")

    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    storage_path = f"sqlite:///{os.path.join(save_dir, 'optuna_study.db')}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        storage=storage_path,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=20)

    # Plotting
    for plot_func, filename in [
        (plot_intermediate_values, "intermediate_values.html"),
        (plot_param_importances, "param_importances.html"),
        (plot_slice, "param_slices.html"),
    ]:
        try:
            fig = plot_func(study)
            fig.write_html(os.path.join(save_dir, filename))
        except Exception as e:
            print(f"[!] Failed to generate {filename}: {e}")

    print("Best hyperparameters:")
    print(study.best_trial.params)

    # Save model
    if best_model:
        model_path = os.path.join(save_dir, best_model_filename)
        torch.save(best_model, model_path)
        print(f"[✓] Best model saved to: {model_path}")

    # Save best params and score
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    with open(os.path.join(save_dir, "best_score.txt"), "w") as f:
        f.write(f"{study.best_value:.4f}")

    export_optuna_results(study, save_dir, best_model_filename, best_test_report, best_train_loss, best_valid_loss)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    set_seed(42)

    # Run Optuna for the best-case scenarios (time vs frequency domain)
    run_optuna_for_combination("chest", "Sc_4_T")
    run_optuna_for_combination("chest", "Sc_4_F")

    # Analyze the logged trials (from log_optuna_trial)
    df = analyze_logged_trials(
        log_csv="optuna_logs/optuna_trials.csv",
        save_dir="optuna_results",
        metric="score"
    )

    # Analyze the saved study database
    df2 = analyze_optuna_study("Sc_4_T", "chest")

    '''
    for position in positions:
        for scenario in scenarios:
            try:
                run_optuna_for_combination(position, scenario)
            except Exception as e:
                print(f"Error in scenario={scenario}, position={position}: {e}")
    '''
    