# training.py
import os
import time
import torch
import torch.nn as nn

from training_imports import show_datasets_info

from data_utils import (
    parse_input,
    collect_datasets_from_input,
    generate_batches,
    create_result_dir,
    set_seed
)

from metrics_utils import get_class_report

from visual_utils import (
    save_loss_curve,
    plot_confusion_matrix,
    get_mlp_feature_importance,
    export_result
)

from training_utils import (
    fit,
    build_model,
    save_model
)
from training_utils import fit, build_model, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    set_seed(42)
    args = parse_input()
    timestamp = str(int(time.time()))
    current_directory = os.path.dirname(__file__)

    results_dir = create_result_dir(current_directory, args.neural_network_type, args.position)
    data_dir = os.path.join(current_directory, "labels_and_data", "data", args.position)
    label_dir = os.path.join(current_directory, "labels_and_data", "labels", args.position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test = collect_datasets_from_input(
        args.position, args.label_type, args.scenario, label_dir, data_dir
    )

    print(show_datasets_info(X_train, y_train, X_val, y_val, X_test, y_test))
    train_dl, val_dl, test_dl = generate_batches(X_train, y_train, X_val, y_val, X_test, y_test)

    model = build_model(args, input_shape, num_labels)

    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)

    print("-" * 90)
    print(model)
    print("-" * 90)

    # Peso para classe positiva (para BCEWithLogitsLoss)
    n_pos = torch.sum(y_train == 1).item()
    n_neg = torch.sum(y_train == 0).item()
    if n_pos == 0:
        print("Erro: Nenhuma amostra da classe positiva no conjunto de treino.")
        return

    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"pos_weight usado: {pos_weight.item():.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    filename = f"{timestamp}_{args.neural_network_type}_{'bin' if num_labels == 2 else 'multi'}_{args.learning_rate}_{args.position}_{args.scenario}"
    checkpoint_path = os.path.join(results_dir, f"{filename}.ckpt")

    model, train_loss, valid_loss = fit(
        args.epochs, args.learning_rate, model, train_dl, val_dl,
        loss_fn, checkpoint_path=checkpoint_path
    )

    if args.export:
        save_loss_curve(train_loss, valid_loss, results_dir, f"{filename}.png")
        print(f"Gráfico de Perda salvo em: {results_dir}")

    report, dict_report, conf_matrix, _, _, auc = get_class_report(model, test_dl, expected_classes=(0, 1))
    print("Relatório de classificação no dataset de teste:")
    print(report)

    if args.export:
        export_result(args.scenario, args.neural_network_type, args.position, dict_report)
        save_model(model, os.path.join("models", f"{filename}.model"))
        print("Relatório e modelo exportados com sucesso.")

        if conf_matrix is not None:
            plot_confusion_matrix(conf_matrix, os.path.join(results_dir, f"{filename}_confusion_matrix.png"))

        if "MLP" in args.neural_network_type.upper():
            get_mlp_feature_importance(model, input_shape, os.path.join(results_dir, f"{filename}_feature_importance.png"))

        if auc:
            with open(os.path.join(results_dir, f"{filename}_auc.txt"), "w") as f:
                f.write(f"AUC-ROC: {auc:.4f}\n")


if __name__ == "__main__":
    main()