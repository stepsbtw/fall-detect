import optuna
import os
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_intermediate_values, plot_param_importances, plot_slice
from utils_data import prepare_datasets
from utils_training import set_seed, build_model_optuna, fit
from utils_output import export_optuna_results, save_report, best_model_optuna

from optuna_analysis import analyze_logged_trials, log_optuna_trial
import gc

from sklearn.metrics import f1_score


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

        log_optuna_trial(trial, f1, save_dir="optuna_logs")
        torch.cuda.empty_cache()
        gc.collect()
        return f1


    print(f"\nRunning Optuna for: scenario={scenario}, position={position}")
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    #study.optimize(objective, n_trials=20, n_jobs=4)
    study.optimize(objective, n_trials=20)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    set_seed(42)
        
    # Rodando optuna somente para a combinação ótima do artigo. Vou comparar dominio do tempo e frequencia.
    run_optuna_for_combination("chest", "Sc_4_T")
    run_optuna_for_combination("chest", "Sc_4_F")

    df = analyze_logged_trials(study_name="caio_optuna", metric="value")


    '''
    for position in positions:
        for scenario in scenarios:
            try:
                run_optuna_for_combination(position, scenario)
            except Exception as e:
                print(f"Error in scenario={scenario}, position={position}: {e}")
    '''
    