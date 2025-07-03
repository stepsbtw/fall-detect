# visual_utils.py
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def save_loss_curve(train_loss, valid_loss, image_dir="./", filename="plot_loss_curve.png"):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(image_dir, filename)
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(cm, path, labels=None):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels is not None else range(cm.shape[0]),
                yticklabels=labels if labels is not None else range(cm.shape[0]))
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def get_mlp_feature_importance(model, input_shape, output_path):
    num_features = input_shape[0] * input_shape[1] if input_shape[1] > 1 else input_shape[0]
    weights = model.layers[0].weight.detach().cpu().numpy().mean(axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), weights)
    plt.title("Feature Importance (First Layer Weights)")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def export_result(scenario, model_type, position, test_report, folder="results"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{scenario}_{model_type}_{position}.json"
    with open(os.path.join(folder, filename), "w") as f:
        json.dump(test_report, f, indent=4)
