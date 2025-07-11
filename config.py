import os
import torch
import numpy as np

class Config:
    """Configurações centralizadas para o projeto de detecção de quedas"""
    
    # ----------------------------- #
    #          Dispositivo          #
    # ----------------------------- #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurações CUDA
    TORCH_BACKENDS = {
        'cudnn_deterministic': False,
        'cudnn_benchmark': True,
        'cuda_matmul_allow_tf32': True,
        'cudnn_allow_tf32': True
    }
    
    # ----------------------------- #
    #   Seeds (Reprodutibilidade)   #
    # ----------------------------- #
    SEED = 42
    
    # ----------------------------- #
    #         Diretórios            #
    # ----------------------------- #
    ROOT_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(ROOT_DIR, "labels_and_data", "data")
    LABEL_PATH = os.path.join(ROOT_DIR, "labels_and_data", "labels")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "output")
    
    # ----------------------------- #
    #         Cenários              #
    # ----------------------------- #
    SCENARIOS = {
        "Sc1_acc_T": ['magacc_time_domain_data_array.npy', (1020, 1)],
        "Sc1_gyr_T": ['maggyr_time_domain_data_array.npy', (1020, 1)],
        "Sc1_acc_F": ['magacc_frequency_domain_data_array.npy', (510, 1)],
        "Sc1_gyr_F": ['maggyr_frequency_domain_data_array.npy', (510, 1)],
        "Sc_2_acc_T": ['acc_x_y_z_axes_time_domain_data_array.npy', (1020, 3)],
        "Sc_2_gyr_T": ['gyr_x_y_z_axes_time_domain_data_array.npy', (1020, 3)],
        "Sc_2_acc_F": ['acc_x_y_z_axes_frequency_domain_data_array.npy', (510, 3)],
        "Sc_2_gyr_F": ['gyr_x_y_z_axes_frequency_domain_data_array.npy', (510, 3)],
        "Sc_3_T": ['magacc_and_maggyr_time_domain_data_array.npy', (1020, 2)],
        "Sc_3_F": ['magacc_and_maggyr_frequency_domain_data_array.npy', (510, 2)],
        "Sc_4_T": ['acc_and_gyr_three_axes_time_domain_data_array.npy', (1020, 6)],
        "Sc_4_F": ['acc_and_gyr_three_axes_frequency_domain_data_array.npy', (510, 6)],
    }
    
    # ----------------------------- #
    #         Labels                #
    # ----------------------------- #
    LABELS_DICT = {
        "multiple_one": "multiple_class_label_1.npy",
        "multiple_two": "multiple_class_label_2.npy",
        "binary_one": "binary_class_label_1.npy",
        "binary_two": "binary_class_label_2.npy",
    }
    
    # ----------------------------- #
    #     Hiperparâmetros Optuna    #
    # ----------------------------- #
    OPTUNA_CONFIG = {
        'n_trials': 50,
        #'timeout': 3600,  # 1 hora
        'n_jobs': 1
    }
    
    # ----------------------------- #
    #     Treinamento              #
    # ----------------------------- #
    TRAINING_CONFIG = {
        'epochs': 25,
        'early_stopping': True,
        'patience': 5,
        'batch_size': 32,
        'num_workers': 8,
        'pin_memory': True,
        'shuffle': True
    }
    
    # ----------------------------- #
    #     Otimização               #
    # ----------------------------- #
    OPTIMIZER_CONFIG = {
        'name': 'Adam',
        'lr_range': (1e-4, 1e-2),
        'lr_log': True
    }
    
    # ----------------------------- #
    #     Cross-Validation         #
    # ----------------------------- #
    CV_CONFIG = {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
    
    # ----------------------------- #
    #     Modelos                  #
    # ----------------------------- #
    MODEL_CONFIGS = {
        'CNN1D': {
            'filter_size_range': (16, 128),
            'kernel_size_range': (3, 7),
            'num_layers_range': (2, 4),
            'num_dense_layers_range': (1, 2),
            'dense_neurons_range': (64, 512)
        },
        'MLP': {
            'num_layers_range': (1, 4),
            'dense_neurons_range': (64, 1024)
        },
        'LSTM': {
            'hidden_dim_range': (64, 256),
            'num_layers_range': (1, 3)
        }
    }
    
    # ----------------------------- #
    #     Métricas                 #
    # ----------------------------- #
    METRICS_CONFIG = {
        'decision_threshold_range': (0.5, 0.9),
        'decision_threshold_step': 0.1,
        'dropout_range': (0.1, 0.5),
        'dropout_step': 0.1
    }
    
    # ----------------------------- #
    #     Data Split               #
    # ----------------------------- #
    DATA_SPLIT = {
        'test_size': 0.2,
        'random_state': 42
    }
    
    # ----------------------------- #
    #     Treinamento Final        #
    # ----------------------------- #
    FINAL_TRAINING = {
        'num_models': 20,
        'seed_offset': 42
    }
    
    @classmethod
    def get_array_size(cls, position):
        """Retorna o tamanho do array baseado na posição"""
        return 1020 if position == "chest" else 450
    
    @classmethod
    def get_num_labels(cls, label_type):
        """Retorna o número de labels baseado no tipo"""
        if label_type == "multiple_one":
            return 37
        elif label_type == "multiple_two":
            return 26
        else:  # binary_one or binary_two
            return 2
    
    @classmethod
    def get_input_shape_dict(cls, scenario, position, model_type=None):
        """Retorna o dicionário de shapes de entrada para cada modelo"""
        array_size = cls.get_array_size(position)
        scenario_config = cls.SCENARIOS[scenario]
        
        # Ajustar shape baseado no array_size
        if position == "chest":
            input_shape = scenario_config[1]
        else:  # left or right
            # Ajustar para 450 samples
            if scenario_config[1][1] == 1:  # 1 feature
                input_shape = (450, 1)
            elif scenario_config[1][1] == 2:  # 2 features
                input_shape = (450, 2)
            elif scenario_config[1][1] == 3:  # 3 features
                input_shape = (450, 3)
            elif scenario_config[1][1] == 6:  # 6 features
                input_shape = (450, 6)
        
        input_shape_dict = {
            "CNN1D": input_shape,
            "MLP": np.prod(input_shape),
            "LSTM": input_shape
        }
        
        return input_shape_dict
    
    @classmethod
    def setup_device(cls):
        """Configura o dispositivo e imprime informações"""
        print("Usando dispositivo:", cls.DEVICE)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Número de GPUs:", torch.cuda.device_count())
            
            if torch.cuda.device_count() > 1:
                print("Configurando para usar múltiplas GPUs...")
                print(f"GPUs disponíveis: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Configurar backends
        torch.backends.cudnn.deterministic = cls.TORCH_BACKENDS['cudnn_deterministic']
        torch.backends.cudnn.benchmark = cls.TORCH_BACKENDS['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32 = cls.TORCH_BACKENDS['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = cls.TORCH_BACKENDS['cudnn_allow_tf32']
    
    @classmethod
    def set_seed(cls, seed=None):
        """Define seeds para reprodutibilidade"""
        if seed is None:
            seed = cls.SEED
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    @classmethod
    def get_output_dir(cls, model_type, position, scenario, label_type):
        """Retorna o diretório de saída"""
        base_out = os.path.join(cls.OUTPUT_PATH, model_type, position, scenario, label_type)
        os.makedirs(base_out, exist_ok=True)
        return base_out 