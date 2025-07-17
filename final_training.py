import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import train, save_results
from config import Config
from utils import load_hyperparameters, load_test_data, create_model 
import numpy as np
from sklearn.model_selection import train_test_split

# --- Remover funções de análise e visualização ---
# Remover: analyze_results, create_visualizations, copy_best_models

def main():
    """Script para treinamento final com os melhores hiperparâmetros"""
    # Configurações
    Config.setup_device()
    Config.set_seed()
    # Argumentos CLI
    parser = argparse.ArgumentParser(description="Treinamento Final com Melhores Hiperparâmetros")
    parser.add_argument("-scenario", required=True, choices=[
        "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
        "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
        "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
    ])
    parser.add_argument("-position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
    parser.add_argument("--nn", required=False, choices=["CNN1D", "MLP", "LSTM"])
    parser.add_argument("--num_models", type=int, default=20, help="Número de modelos para treinar")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas")
    args = parser.parse_args()
    # Parâmetros
    position = args.position
    label_type = args.label_type
    scenario = args.scenario
    model_type_arg = args.nn
    num_models = args.num_models
    epochs = args.epochs
    num_labels = Config.get_num_labels(label_type)
    # Diretórios
    base_out = Config.get_output_dir(model_type_arg, position, scenario, label_type)
    os.makedirs(base_out, exist_ok=True)
    # Carregar hiperparâmetros
    best_params = load_hyperparameters(base_out)
    if "best_params" in best_params:
        best_params = best_params["best_params"]
    model_type = best_params["model_type"] if "model_type" in best_params else model_type_arg
    # Carregar dados de treino/validação/teste
    data = np.load(os.path.join(base_out, "test_data.npz"))
    X_trainval, y_trainval = data['X_trainval'], data['y_trainval']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Dividir X_trainval em treino e validação para os 20treinos finais
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=Config.DATA_SPLIT['test_size'],
        random_state=Config.DATA_SPLIT['random_state']
    )
    
    input_shape_dict = Config.get_input_shape_dict(scenario, position, model_type)
    input_shape = input_shape_dict[model_type]
    # Treinamento dos modelos finais
    for i in range(1, num_models +1):
        print(f"\nTreinando modelo final {i}/{num_models}...")
        model = create_model(model_type, best_params, input_shape, num_labels)
        model.to(Config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        batch_size = Config.TRAINING_CONFIG.get('batch_size', 32)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        # Treinar
        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, val_loader, optimizer, criterion, Config.DEVICE,
            epochs=epochs, early_stopping=False, patience=5, scaler=None
        )
        # Salvar modelo
        model_dir = os.path.join(base_out, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_{i}.pt"))
        
        # Criar loader para dados de teste para avaliação final
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        # Salvar métricas brutas usando dados de teste
        save_results(
            model=model,
            val_loader=test_loader,
            y_val_onehot=y_test,
            number_of_labels=num_labels,
            i=i,
            decision_threshold=best_params.get("decision_threshold", 0.5),
            output_dir=model_dir,
            device=Config.DEVICE
        )
        print(f"Modelo {i} treinado e salvo em {model_dir}")
    print(f"\nTreinamento final concluído! Resultados salvos em: {base_out}")

# --- Remover chamada para análise automática, visualizações e cópia de modelos ---
if __name__ == "__main__":
    main() 