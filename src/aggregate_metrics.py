import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

def aggregate_model_metrics(base_out):
    """
    Agrega métricas dos modelos individuais em arquivos consolidados
    """
    print(f"\n{'='*50}")
    print("AGREGANDO MÉTRICAS DOS MODELOS FINAIS")
    print(f"{'='*50}")
    
    # Lista para armazenar todas as métricas
    all_metrics = []
    
    # Procurar por arquivos de métricas dos modelos
    for i in range(1, 21):  # model_1 até model_20
        model_dir = os.path.join(base_out, f"model_{i}")
        metrics_file = os.path.join(model_dir, f"metrics_model_{i}.csv")
        
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                all_metrics.append(df)
                print(f"Modelo {i}: MCC={df['MCC'].iloc[0]:.4f}, Acc={df['Accuracy'].iloc[0]:.4f}")
            except Exception as e:
                print(f"Erro ao ler métricas do modelo {i}: {e}")
        else:
            print(f"Arquivo de métricas não encontrado para modelo {i}: {metrics_file}")
    
    if not all_metrics:
        print("Nenhuma métrica encontrada!")
        return False
    
    # Combinar todas as métricas
    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    # Garantir que a coluna Model está no formato correto (float)
    combined_df['Model'] = combined_df['Model'].astype(float)
    
    # Reordenar colunas para o formato esperado
    expected_columns = ['Model', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'tp', 'tn', 'fp', 'fn']
    available_columns = [col for col in expected_columns if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    # Salvar all_metrics.csv
    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    combined_df.to_csv(all_metrics_path, index=False)
    print(f"\nMétricas consolidadas salvas em: {all_metrics_path}")
    print(f"Total de modelos processados: {len(combined_df)}")
    
    # Criar summary_metrics.csv no formato esperado
    # Calcular estatísticas apenas para as colunas numéricas (excluindo Model)
    numeric_columns = [col for col in combined_df.columns if col != 'Model']
    summary_stats = combined_df[numeric_columns].describe()
    
    # Criar DataFrame no formato esperado (com mean e std como index)
    summary_df = summary_stats.loc[['mean', 'std']].copy()
    summary_df.index.name = None  # Remove o nome do index
    
    # Adicionar coluna Model com valores vazios para mean e std
    summary_df.insert(0, 'Model', ['mean', 'std'])
    
    summary_path = os.path.join(base_out, "summary_metrics.csv")
    summary_df.to_csv(summary_path)
    print(f"Estatísticas resumidas salvas em: {summary_path}")
    
    # Criar visualizações
    create_visualizations(combined_df, base_out)
    
    return True

def create_visualizations(df, base_out):
    """
    Cria visualizações das métricas agregadas
    """
    print("\nCriando visualizações...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Boxplot das métricas
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_cols = [col for col in ['MCC', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity'] 
                   if col in df.columns]
    
    if metrics_cols:
        df[metrics_cols].boxplot(ax=ax)
        ax.set_title('Distribuição das Métricas dos Modelos Finais', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor da Métrica', fontsize=12)
        ax.set_xlabel('Métrica', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        boxplot_path = os.path.join(base_out, "metrics_boxplot.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot salvo em: {boxplot_path}")
    
    # 2. Histograma do MCC
    if 'MCC' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['MCC'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(df['MCC'].mean(), color='red', linestyle='--', 
                   label=f'Média: {df["MCC"].mean():.4f}')
        plt.axvline(df['MCC'].median(), color='green', linestyle='--', 
                   label=f'Mediana: {df["MCC"].median():.4f}')
        plt.xlabel('MCC', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição do MCC dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        mcc_hist_path = os.path.join(base_out, "mcc_histogram.png")
        plt.savefig(mcc_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histograma MCC salvo em: {mcc_hist_path}")
    
    # 3. Scatter plot MCC vs Accuracy
    if 'MCC' in df.columns and 'Accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Accuracy'], df['MCC'], alpha=0.7, s=50)
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('MCC', fontsize=12)
        plt.title('MCC vs Accuracy dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(base_out, "mcc_vs_accuracy.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot salvo em: {scatter_path}")
    
    # 4. Matriz de correlação
    if len(metrics_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[metrics_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Matriz de Correlação das Métricas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        corr_path = os.path.join(base_out, "correlation_heatmap.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Matriz de correlação salva em: {corr_path}")
    
    # 5. Estatísticas detalhadas
    print("\nEstatísticas das métricas:")
    for col in metrics_cols:
        print(f"{col}:")
        print(f"  Média: {df[col].mean():.4f}")
        print(f"  Desvio: {df[col].std():.4f}")
        print(f"  Mín: {df[col].min():.4f}")
        print(f"  Máx: {df[col].max():.4f}")
        print(f"  Mediana: {df[col].median():.4f}")
        print()
    
    # Mostrar estatísticas no formato do summary_metrics.csv
    print("Formato do summary_metrics.csv:")
    numeric_columns = [col for col in df.columns if col != 'Model']
    summary_stats = df[numeric_columns].describe()
    summary_df = summary_stats.loc[['mean', 'std']].copy()
    summary_df.insert(0, 'Model', ['mean', 'std'])
    print(summary_df.to_string())

def main():
    """Script para agregar métricas dos modelos finais"""
    parser = argparse.ArgumentParser(description="Agregar métricas dos modelos finais")
    parser.add_argument("-scenario", required=True, choices=[
        "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
        "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
        "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
    ])
    parser.add_argument("-position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
    parser.add_argument("--nn", required=True, choices=["CNN1D", "MLP", "LSTM"])
    
    args = parser.parse_args()
    
    # Configurar diretório de saída
    base_out = Config.get_output_dir(args.nn, args.position, args.scenario, args.label_type)
    
    print(f"Diretório de saída: {base_out}")
    
    # Verificar se o diretório existe
    if not os.path.exists(base_out):
        print(f"Erro: Diretório não encontrado: {base_out}")
        print("Execute o treinamento final primeiro.")
        return
    
    # Agregar métricas
    success = aggregate_model_metrics(base_out)
    
    if success:
        print(f"\n{'='*50}")
        print("AGREGAÇÃO DE MÉTRICAS CONCLUÍDA COM SUCESSO!")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("ERRO NA AGREGAÇÃO DE MÉTRICAS!")
        print(f"{'='*50}")

if __name__ == "__main__":
    main() 