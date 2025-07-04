import os
import json
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def save_loss_curve(train_loss: list, valid_loss: list, image_dir: str = "./", filename: str = "plot_loss_curve"):
    fig = plt.figure(figsize=(10, 8))

    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(image_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_report(model, test_dl, expected_classes=(0,1)):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_dl:
            device = next(model.parameters()).device
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs.float())
            probs = torch.sigmoid(outputs.squeeze())
            threshold = getattr(model, 'decision_threshold', 0.5)
            preds = (probs >= threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_predictions, zero_division=0)
    dict_report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None

    try:
        for cls in expected_classes:
            str_cls = str(cls)
            if str_cls not in dict_report:
                dict_report[str_cls] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 0
                }
    except Exception as e:
        print(f"Erro ao gerar classification_report: {e}")
        report = "Erro ao gerar classification_report"
        dict_report = {str(cls): {"f1-score": 0.0} for cls in expected_classes}

    return report, dict_report, conf_matrix, all_labels, all_probs, all_predictions, auc


def export_optuna_results(study, save_dir, best_test_report, best_train_loss, best_valid_loss):
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

def best_model_optuna(model, model_type, trial, f1, input_shape, test_dl, save_dir):
    model_filename = f"best_{model_type}_trial_{trial.number}_f1_{f1:.4f}.model"
    model_path = os.path.join(save_dir, model_filename)

    torch.save(model, model_path)
    print(f"[Trial {trial.number}] Novo melhor modelo salvo: {model_path}")

    report, dict_report, conf_matrix, all_labels, all_probs, auc = save_report(model, test_dl, expected_classes=(0, 1))

    return model_filename, dict_report

def export_result(scenario, model_type, position, test_report, folder="results"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{scenario}_{model_type}_{position}.json"
    with open(os.path.join(folder, filename), "w") as f:
        json.dump(test_report, f, indent=4)

def create_result_dir(base_dir, model_type, pos):
    result_dir = os.path.join(base_dir, "output", model_type, pos)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir