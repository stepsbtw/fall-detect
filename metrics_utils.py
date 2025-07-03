from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_report(model, test_dl, expected_classes=(0, 1)):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs >= 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Relatórios
    try:
        report = classification_report(all_labels, all_predictions, zero_division=0)
        dict_report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)

        # Força entrada de todas as classes esperadas
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

    try:
        cm = confusion_matrix(all_labels, all_predictions, labels=expected_classes)
    except Exception as e:
        print(f"Erro ao gerar confusion_matrix: {e}")
        cm = None

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = None

    return report, dict_report, cm, all_labels, all_probs, auc
