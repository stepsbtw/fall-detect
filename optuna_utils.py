# optuna_utils.py
import os
import torch
import json
import pandas as pd
from training_imports import (
    get_class_report, plot_confusion_matrix,
    get_mlp_feature_importance, save_loss_curve
)
from training import save_model

def handle_best_model(model, model_type, trial, f1, input_shape, test_dl, save_dir):
    model_filename = f"best_{model_type}_trial_{trial.number}_f1_{f1:.4f}.model"
    model_path = os.path.join(save_dir, model_filename)

    save_model(model, model_path)
    print(f"[Trial {trial.number}] Novo melhor modelo salvo: {model_path}")

    _, test_report, cm, _, _, auc = get_class_report(model, test_dl)

    if cm is not None:
        plot_confusion_matrix(cm, os.path.join(save_dir, "best_model_confusion_matrix.png"))

    if model_type == "MLP":
        get_mlp_feature_importance(model, input_shape, os.path.join(save_dir, "best_model_feature_importance.png"))

    if auc:
        with open(os.path.join(save_dir, "best_model_auc.txt"), "w") as f:
            f.write(f"AUC-ROC: {auc:.4f}\n")

    return model_filename, test_report


def export_optuna_results(study, save_dir, best_model_filename, best_test_report, best_train_loss, best_valid_loss):
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(save_dir, "best_config.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    if best_test_report:
        with open(os.path.join(save_dir, "best_model_test_report.json"), "w") as f:
            json.dump(best_test_report, f, indent=4)

    if best_train_loss and best_valid_loss:
        save_loss_curve(best_train_loss, best_valid_loss, image_dir=save_dir, filename="best_model_loss_curve.png")

    print("Todos os resultados exportados com sucesso.")
