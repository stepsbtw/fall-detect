import os
import optuna
import json
import optuna.visualization as vis
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config

def load_optuna(output_dir, study_name):

    db_path = os.path.join(output_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    # Tenta carregar o estudo se já existe, senão cria novo
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"Estudo existente carregado de: {db_path}")

    print("Melhor MCC:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)

    # Salvar resultados
    os.makedirs(output_dir, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    # Importância dos hiperparâmetros
    try:
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, "param_importance.png"))
    except Exception as e:
        print(f"Could not save importance plot: {e}")

    return study

def main():
    """Script para busca de hiperparâmetros usando Optuna"""
    
    # ----------------------------- #
    #         Configurações         #
    # ----------------------------- #
    Config.setup_device()
    Config.set_seed()
    
    # ----------------------------- #
    #         Argumentos CLI        #
    # ----------------------------- #
    parser = argparse.ArgumentParser(description="Busca de Hiperparâmetros com Optuna")
    
    parser.add_argument("-scenario", required=True, choices=[
        "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
        "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
        "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
    ])
    parser.add_argument("-position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
    parser.add_argument("--nn", required=False, choices=["CNN1D", "MLP", "LSTM"])
    parser.add_argument("--n_trials", type=int, default=50, help="Número de trials para Optuna")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout em segundos")
    
    args = parser.parse_args()
    
    # ----------------------------- #
    #         Parâmetros            #
    # ----------------------------- #
    position = args.position
    label_type = args.label_type
    scenario = args.scenario
    model_type_arg = args.nn
    num_labels = Config.get_num_labels(label_type)
    
    # Atualizar configurações do Optuna
    Config.OPTUNA_CONFIG['n_trials'] = args.n_trials
    Config.OPTUNA_CONFIG['timeout'] = args.timeout
    
    # ----------------------------- #
    #         Diretórios            #
    # ----------------------------- #
    data_path = os.path.join(Config.DATA_PATH, position)
    label_path = os.path.join(Config.LABEL_PATH, position)
    base_out = Config.get_output_dir(model_type_arg, position, scenario, label_type)
    
    # ----------------------------- #
    #     Carregar Dados e Split    #
    # ----------------------------- #
    print(f"\nCarregando dados para cenário: {scenario}")
    print(f"Posição: {position}")
    print(f"Tipo de label: {label_type}")
    print(f"Modelo: {model_type_arg if model_type_arg else 'Todos'}")
    
    # Carregar dados
    scenario_config = Config.SCENARIOS[scenario]
    X = np.load(os.path.join(data_path, scenario_config[0]))
    y = np.load(os.path.join(label_path, Config.LABELS_DICT[label_type])).astype(np.int64)
    
    if model_type_arg == "LSTM":
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], -1, 1))
    
    print(f"Shape dos dados: {X.shape}")
    print(f"Shape dos labels: {y.shape}")
    
    # Split dos dados
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, 
        test_size=Config.DATA_SPLIT['test_size'], 
        random_state=Config.DATA_SPLIT['random_state']
        shuffle=False
    )
    
    print(f"Train/Val split: {X_trainval.shape[0]} amostras")
    print(f"Test split: {X_test.shape[0]} amostras")
    
    # ----------------------------- #
    #     Preparar Input Shapes     #
    # ----------------------------- #
    input_shape_dict = Config.get_input_shape_dict(scenario, position, model_type_arg)
    print(f"Input shapes: {input_shape_dict}")
    
    # ----------------------------- #
    #     Otimização com Optuna     #
    # ----------------------------- #
    print(f"\nIniciando Otimização com Optuna...")
    print(f"Número de trials: {Config.OPTUNA_CONFIG['n_trials']}")
    print(f"Timeout: {Config.OPTUNA_CONFIG['timeout']} segundos")
    
    study_name = f"{scenario}_{position}_{label_type}_{model_type_arg}" if model_type_arg else f"{scenario}_{position}_{label_type}"
    
    study = load_optuna(output_dir=base_out, study_name=study_name)
    
    # ----------------------------- #
    #     Salvar Resultados         #
    # ----------------------------- #
    best_params = study.best_params
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg
    
    print(f"\n{'='*50}")
    print(f"MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print(f"{'='*50}")
    print(f"Modelo: {model_type}")
    print(f"Melhor valor: {study.best_value:.4f}")
    print(f"Parâmetros:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Salvar melhores parâmetros
    results_file = os.path.join(base_out, "best_hyperparameters.json")
    results = {
        "scenario": scenario,
        "position": position,
        "label_type": label_type,
        "model_type": model_type,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials),
        "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados salvos em: {results_file}")
    
    # Salvar dados de treino/validação e teste para treinamento final
    test_data_file = os.path.join(base_out, "test_data.npz")
    np.savez(test_data_file, X_trainval=X_trainval, y_trainval=y_trainval, X_test=X_test, y_test=y_test)
    print(f"Dados de treino/validação e teste salvos em: {test_data_file}")
    
    print(f"\nBusca de hiperparâmetros concluída!")
    print(f"Próximo passo: executar treinamento final com os melhores parâmetros")

if __name__ == "__main__":
    main() 