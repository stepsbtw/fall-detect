import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neural_networks import CNN1DNet, MLPNet, LSTMNet
from utils import train, save_results, plot_loss_curve
from config import Config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def load_hyperparameters(output_dir):
    """Carrega os melhores hiperparâmetros encontrados"""
    results_file = os.path.join(output_dir, "best_hyperparameters.json")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Arquivo de hiperparâmetros não encontrado: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def load_test_data(output_dir):
    """Carrega os dados de teste salvos"""
    test_data_file = os.path.join(output_dir, "test_data.npz")
    
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"Arquivo de dados de teste não encontrado: {test_data_file}")
    
    data = np.load(test_data_file)
    return data['X_test'], data['y_test']

def create_model(model_type, best_params, input_shape, num_labels):
    """Cria o modelo com os melhores hiperparâmetros"""
    if model_type == "CNN1D":
        model = CNN1DNet(
            input_shape=input_shape,
            filter_size=best_params["filter_size"],
            kernel_size=best_params["kernel_size"],
            num_layers=best_params["num_layers"],
            num_dense_layers=best_params["num_dense_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    elif model_type == "MLP":
        model = MLPNet(
            input_dim=input_shape,
            num_layers=best_params["num_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    elif model_type == "LSTM":
        model = LSTMNet(
            input_dim=input_shape[1],
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    return model

def analyze_results(output_dir, num_models=20):
    """Analisa os resultados dos modelos treinados"""
    
    print(f"\n{'='*50}")
    print(f"ANÁLISE DOS RESULTADOS")
    print(f"{'='*50}")
    
    # Carregar hiperparâmetros
    results = load_hyperparameters(output_dir)
    print(f"Modelo: {results['model_type']}")
    print(f"Melhor valor encontrado: {results['best_value']:.4f}")
    
    # Criar lista para armazenar métricas
    all_metrics = []
    
    # Carregar métricas de todos os modelos
    for i in range(1, num_models + 1):
        metrics_path = os.path.join(output_dir, f"model_{i}", f"metrics_model_{i}.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            all_metrics.append(df.iloc[0])
            print(f"Modelo {i}: MCC = {df.iloc[0]['MCC']:.4f}")
        else:
            print(f"Arquivo de métricas não encontrado para modelo {i}")
    
    if not all_metrics:
        print("Nenhuma métrica encontrada!")
        return
    
    # Gerar DataFrame com todas as métricas
    metrics_df = pd.DataFrame(all_metrics)
    
    # Salvar todas as métricas em CSV
    all_metrics_path = os.path.join(output_dir, "all_metrics.csv")
    metrics_df.to_csv(all_metrics_path, index=False)
    print(f"\nTodas as métricas salvas em: {all_metrics_path}")
    
    # Calcular estatísticas
    summary = metrics_df.describe().loc[["mean", "std"]]
    
    # Salvar resumo estatístico
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary.to_csv(summary_path)
    print(f"Resumo estatístico salvo em: {summary_path}")
    
    # Imprimir estatísticas
    print(f"\n{'='*50}")
    print(f"ESTATÍSTICAS DOS MODELOS")
    print(f"{'='*50}")
    print(f"Número de modelos analisados: {len(metrics_df)}")
    print(f"\nMétricas principais:")
    print(f"MCC - Média: {metrics_df['MCC'].mean():.4f}, Std: {metrics_df['MCC'].std():.4f}")
    print(f"Accuracy - Média: {metrics_df['Accuracy'].mean():.4f}, Std: {metrics_df['Accuracy'].std():.4f}")
    print(f"Precision - Média: {metrics_df['Precision'].mean():.4f}, Std: {metrics_df['Precision'].std():.4f}")
    print(f"Sensitivity - Média: {metrics_df['Sensitivity'].mean():.4f}, Std: {metrics_df['Sensitivity'].std():.4f}")
    print(f"Specificity - Média: {metrics_df['Specificity'].mean():.4f}, Std: {metrics_df['Specificity'].std():.4f}")
    
    # Encontrar melhor modelo
    best_model_idx = metrics_df['MCC'].idxmax()
    best_model_num = best_model_idx + 1
    best_mcc = metrics_df.loc[best_model_idx, 'MCC']
    
    print(f"\nMelhor modelo (baseado em MCC):")
    print(f"Modelo {best_model_num}: MCC = {best_mcc:.4f}")
    
    # Criar visualizações
    create_visualizations(metrics_df, output_dir)
    
    # Copiar melhores modelos
    copy_best_models(output_dir, num_models)
    
    print(f"\nAnálise concluída! Resultados salvos em: {output_dir}")

def create_visualizations(metrics_df, output_dir):
    """Cria visualizações dos resultados"""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Boxplot das principais métricas
    plt.figure(figsize=(12, 8))
    metricas_plot = ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity"]
    sns.boxplot(data=metrics_df[metricas_plot])
    plt.title("Distribuição das Métricas dos Modelos", fontsize=14, fontweight='bold')
    plt.ylabel("Valor da Métrica")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histograma do MCC
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['MCC'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(metrics_df['MCC'].mean(), color='red', linestyle='--', 
                label=f'Média: {metrics_df["MCC"].mean():.4f}')
    plt.xlabel('MCC')
    plt.ylabel('Frequência')
    plt.title('Distribuição do MCC dos Modelos', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcc_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot MCC vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Accuracy'], metrics_df['MCC'], alpha=0.7, s=50)
    plt.xlabel('Accuracy')
    plt.ylabel('MCC')
    plt.title('MCC vs Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcc_vs_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap de correlação
    plt.figure(figsize=(10, 8))
    correlation_matrix = metrics_df[metricas_plot].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlação das Métricas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizações criadas em: {output_dir}")

def copy_best_models(output_dir, num_models):
    """Copia os melhores modelos para um diretório separado"""
    
    best_dir = os.path.join(output_dir, "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    copied_count = 0
    for i in range(1, num_models + 1):
        src = os.path.join(output_dir, f"model_{i}", f"model_{i}.pt")
        if os.path.exists(src):
            dst = os.path.join(best_dir, f"model_{i}.pt")
            shutil.copyfile(src, dst)
            copied_count += 1
    
    print(f"Copiados {copied_count} modelos para: {best_dir}")

def main():
    """Script para treinamento final com os melhores hiperparâmetros"""
    
    # ----------------------------- #
    #         Configurações         #
    # ----------------------------- #
    Config.setup_device()
    Config.set_seed()
    
    # ----------------------------- #
    #         Argumentos CLI        #
    # ----------------------------- #
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
    parser.add_argument("--no_analysis", action="store_true", help="Pular análise de resultados")
    
    args = parser.parse_args()
    
    # ----------------------------- #
    #         Parâmetros            #
    # ----------------------------- #
    position = args.position
    label_type = args.label_type
    scenario = args.scenario
    model_type_arg = args.nn
    num_labels = Config.get_num_labels(label_type)
    
    # Atualizar configurações de treinamento
    Config.TRAINING_CONFIG['epochs'] = args.epochs
    Config.FINAL_TRAINING['num_models'] = args.num_models
    
    # ----------------------------- #
    #         Diretórios            #
    # ----------------------------- #
    base_out = Config.get_output_dir(model_type_arg, position, scenario, label_type)
    
    # ----------------------------- #
    #   Carregar Hiperparâmetros   #
    # ----------------------------- #
    print(f"\nCarregando melhores hiperparâmetros...")
    results = load_hyperparameters(base_out)
    
    best_params = results["best_params"]
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg
    
    print(f"Modelo: {model_type}")
    print(f"Melhor valor encontrado: {results['best_value']:.4f}")
    print(f"Parâmetros:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # ----------------------------- #
    #     Carregar Dados            #
    # ----------------------------- #
    print(f"\nCarregando dados...")
    
    # Carregar dados de treino/val
    data_path = os.path.join(Config.DATA_PATH, position)
    label_path = os.path.join(Config.LABEL_PATH, position)
    
    scenario_config = Config.SCENARIOS[scenario]
    X = np.load(os.path.join(data_path, scenario_config[0]))
    y = np.load(os.path.join(label_path, Config.LABELS_DICT[label_type])).astype(np.int64)
    
    if model_type == "LSTM":
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], -1, 1))
    
    # Carregar dados de teste salvos
    X_test, y_test = load_test_data(base_out)
    
    print(f"Shape dos dados completos: {X.shape}")
    print(f"Shape dos dados de teste: {X_test.shape}")
    
    # ----------------------------- #
    #     Preparar Input Shape      #
    # ----------------------------- #
    input_shape_dict = Config.get_input_shape_dict(scenario, position, model_type)
    
    if model_type == "CNN1D":
        input_shape = input_shape_dict["CNN1D"]
    elif model_type == "LSTM":
        input_shape = X.shape[1:]
    else:  # MLP
        input_shape = input_shape_dict["MLP"]
    
    print(f"Input shape para {model_type}: {input_shape}")
    
    # ----------------------------- #
    #     Treinamento Final         #
    # ----------------------------- #
    print(f"\nIniciando Treinamento Final dos {Config.FINAL_TRAINING['num_models']} Modelos...")
    
    batch_size = Config.TRAINING_CONFIG['batch_size']
    
    # Usar múltiplas GPUs se disponível
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
        batch_size = batch_size * torch.cuda.device_count()
        print(f"Batch size ajustado para {batch_size}")
    
    # Preparar data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), 
        batch_size=batch_size, 
        shuffle=Config.TRAINING_CONFIG['shuffle'], 
        pin_memory=Config.TRAINING_CONFIG['pin_memory'], 
        num_workers=Config.TRAINING_CONFIG['num_workers']
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ), 
        batch_size=batch_size, 
        pin_memory=Config.TRAINING_CONFIG['pin_memory'], 
        num_workers=Config.TRAINING_CONFIG['num_workers']
    )
    
    # Treinar múltiplos modelos
    for i in range(1, Config.FINAL_TRAINING['num_models'] + 1):
        print(f"\n{'='*50}")
        print(f"TREINANDO MODELO {i}/{Config.FINAL_TRAINING['num_models']}")
        print(f"{'='*50}")
        
        # Definir seed para reprodutibilidade
        seed = Config.FINAL_TRAINING['seed_offset'] + i
        Config.set_seed(seed)
        
        # Criar modelo
        model = create_model(model_type, best_params, input_shape, num_labels)
        model.to(Config.DEVICE, non_blocking=True)
        
        # Usar múltiplas GPUs se disponível
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # Configurar otimizador e loss
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        criterion = nn.CrossEntropyLoss()
        
        # Configurar mixed precision se disponível
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Treinar modelo
        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, test_loader,
            optimizer, criterion, Config.DEVICE,
            epochs=Config.TRAINING_CONFIG['epochs'],
            early_stopping=Config.TRAINING_CONFIG['early_stopping'],
            patience=Config.TRAINING_CONFIG['patience'],
            scaler=scaler
        )
        
        # Salvar resultados
        model_dir = os.path.join(base_out, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Plotar curva de loss
        plot_loss_curve(train_losses, val_losses, model_dir, i)
        
        # Salvar métricas
        save_results(
            model=model,
            val_loader=test_loader,
            y_val_onehot=y_test,
            number_of_labels=num_labels,
            i=i,
            decision_threshold=best_params["decision_threshold"],
            output_dir=model_dir,
            device=Config.DEVICE
        )
        
        # Limpar memória
        del model
        del optimizer
        torch.cuda.empty_cache()
        
        print(f"Modelo {i} treinado e salvo em: {model_dir}")
    
    print(f"\n{'='*50}")
    print(f"TREINAMENTO FINAL CONCLUÍDO!")
    print(f"{'='*50}")
    print(f"Todos os {Config.FINAL_TRAINING['num_models']} modelos foram treinados")
    
    # ----------------------------- #
    #     Análise de Resultados     #
    # ----------------------------- #
    if not args.no_analysis:
        print(f"\n{'='*50}")
        print(f"INICIANDO ANÁLISE DOS RESULTADOS")
        print(f"{'='*50}")
        
        analyze_results(base_out, Config.FINAL_TRAINING['num_models'])
    else:
        print(f"\nAnálise de resultados pulada (--no_analysis)")
    
    print(f"\n{'='*50}")
    print(f"PROCESSO COMPLETO CONCLUÍDO!")
    print(f"{'='*50}")
    print(f"Resultados salvos em: {base_out}")

if __name__ == "__main__":
    main() 