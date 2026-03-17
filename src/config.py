import os
import torch
import numpy as np

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_BACKENDS = {
        'cudnn_deterministic': True,
        'cudnn_benchmark': False,
        # 'cudnn_deterministic': False,
        # 'cudnn_benchmark': True,
        'cuda_matmul_allow_tf32': True,
        'cudnn_allow_tf32': True
    }
    SEED = 42
    ROOT_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.normpath(os.path.join(ROOT_DIR, "..", "dataset"))
    NUM_LABELS = 2

    # Models that use sklearn/XGBoost/CatBoost instead of PyTorch
    CLASSICAL_MODELS = frozenset({"RF", "SVM", "XGBoost", "CatBoost"})

    # [directory_name, filename, (seq_len, num_features)]
    SCENARIOS = {
        #"old_chest_T":            ["chest",            "data_time_domain.npy",      (1100, 8)],
        "chest_T":            ["chest",            "data_time_domain.npy",      (460,  8)],
        # "chest_F":            ["chest",            "data_frequency_domain.npy", (550,  8)],
        "left_T":             ["left",             "data_time_domain.npy",      (460,  8)],
        # "left_F":             ["left",             "data_frequency_domain.npy", (230,  8)],
        "right_T":            ["right",            "data_time_domain.npy",      (460,  8)],
        # "right_F":            ["right",            "data_frequency_domain.npy", (230,  8)],
        "chest_left_T":       ["chest_left",       "data_time_domain.npy",      (460, 16)],
        # "chest_left_F":       ["chest_left",       "data_frequency_domain.npy", (230, 16)],
        "chest_right_T":      ["chest_right",      "data_time_domain.npy",      (460, 16)],
        # "chest_right_F":      ["chest_right",      "data_frequency_domain.npy", (230, 16)],
        "chest_left_right_T": ["chest_left_right", "data_time_domain.npy",      (460, 24)],
        # "chest_left_right_F": ["chest_left_right", "data_frequency_domain.npy", (230, 24)],
    }

    OPTUNA_CONFIG = {
        'n_trials': 30,
        'n_jobs': 1,
    }
    
    TRAINING_CONFIG = {
        'epochs': 500,
        'early_stopping': True,
        'patience': 30,
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': True,
    }

    FINAL_TRAINING = {
        'seed_offset': 0,
    }

    OPTIMIZER_CONFIG = {
        'name': 'Adam',
        'lr_range': (1e-4, 1e-2),
        'lr_log': True
    }

    MODEL_CONFIGS = {
        'CNN1D': {
            'num_layers_range':       (1,  4),
            'filter_size_range':      (16, 128),
            'kernel_size_range':      (3,  9),
            'num_dense_layers_range': (1,  3),
            'dense_neurons_range':    (32, 256),
        },
        'MLP': {
            'num_layers_range':    (1,  5),
            'dense_neurons_range': (32, 512),
        },
        'LSTM': {
            'num_layers_range': (1,  3),
            'hidden_dim_range': (32, 256),
        },
        'GRU': {
            'num_layers_range': (1,  3),
            'hidden_dim_range': (32, 256),
        },
        'RF': {
            'n_estimators_range':      (50, 500),
            'max_depth_range':         (3,  20),
            'min_samples_split_range': (2,  20),
        },
        'SVM': {
            'C_range': (0.01, 100.0),
        },
        'XGBoost': {
            'n_estimators_range':     (50,   500),
            'max_depth_range':        (3,    10),
            'learning_rate_range':    (0.01, 0.3),
            'subsample_range':        (0.6,  1.0),
            'colsample_bytree_range': (0.6,  1.0),
        },
        'CatBoost': {
            'n_estimators_range':  (50,   500),
            'depth_range':         (3,    10),
            'learning_rate_range': (0.01, 0.3),
            'l2_leaf_reg_range':   (1.0,  10.0),
        },
    }
    
    METRICS_CONFIG = {
        'fall_class': 1,                          # label index that represents a fall
        'decision_threshold_range': (0.3, 0.7),  # lower range to improve fall recall
        'decision_threshold_step': 0.05,
        'dropout_range': (0.1, 0.5),
        'dropout_step': 0.1
    }
    
    N_INDIVIDUALS = 15       # total number of subjects in the dataset

    LEARNING_CURVE_CONFIG = {
        'fractions': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'epochs': 10
    }

    DEFAULT_PARAMS = {
        'CNN1D': {
            'model_type':        'CNN1D',
            'learning_rate':     1e-3,
            'dropout':           0.3,
            'decision_threshold': 0.5,
            'filter_size':       64,
            'kernel_size':       5,
            'num_layers':        2,
            'num_dense_layers':  1,
            'dense_neurons':     128,
        },
        'MLP': {
            'model_type':        'MLP',
            'learning_rate':     1e-3,
            'dropout':           0.3,
            'decision_threshold': 0.5,
            'num_layers':        2,
            'dense_neurons':     128,
        },
        'LSTM': {
            'model_type':        'LSTM',
            'learning_rate':     1e-3,
            'dropout':           0.3,
            'decision_threshold': 0.5,
            'hidden_dim':        64,
            'num_layers':        2,
        },
        'GRU': {
            'model_type':        'GRU',
            'learning_rate':     1e-3,
            'dropout':           0.3,
            'decision_threshold': 0.5,
            'hidden_dim':        64,
            'num_layers':        2,
        },
        'RF': {
            'model_type':        'RF',
            'decision_threshold': 0.5,
            'n_estimators':      200,
            'max_depth':         10,
            'min_samples_split': 5,
        },
        'SVM': {
            'model_type':        'SVM',
            'decision_threshold': 0.5,
            'C':                 1.0,
        },
        'XGBoost': {
            'model_type':        'XGBoost',
            'decision_threshold': 0.5,
            'n_estimators':      200,
            'max_depth':         5,
            'learning_rate':     0.1,
            'subsample':         0.8,
            'colsample_bytree':  0.8,
        },
        'CatBoost': {
            'model_type':        'CatBoost',
            'decision_threshold': 0.5,
            'n_estimators':      200,
            'depth':             6,
            'learning_rate':     0.1,
            'l2_leaf_reg':       3.0,
        },
    }
    
    @classmethod
    def get_groups_file(cls, scenario):
        dir_name = cls.SCENARIOS[scenario][0]
        return os.path.normpath(os.path.join(cls.DATA_PATH, dir_name, "labels", "groups.npy"))

    @classmethod
    def get_data_file(cls, scenario):
        dir_name, filename, _ = cls.SCENARIOS[scenario]
        return os.path.normpath(os.path.join(cls.DATA_PATH, dir_name, "data", filename))

    @classmethod
    def get_labels_file(cls, scenario):
        dir_name = cls.SCENARIOS[scenario][0]
        return os.path.normpath(os.path.join(cls.DATA_PATH, dir_name, "labels", "labels.npy"))

    @classmethod
    def get_output_dir(cls, model_type, scenario):
        base = os.path.normpath(os.path.join(cls.ROOT_DIR, "..", "output"))
        if model_type:
            return os.path.join(base, model_type, scenario)
        return os.path.join(base, scenario)

    @classmethod
    def get_models_dir(cls, model_type, scenario):
        base = os.path.normpath(os.path.join(cls.ROOT_DIR, "..", "models"))
        if model_type:
            return os.path.join(base, model_type, scenario)
        return os.path.join(base, scenario)

    @classmethod
    def get_input_shape_dict(cls, scenario, model_type=None):
        seq_len, num_features = cls.SCENARIOS[scenario][2]
        flat = seq_len * num_features
        full = {
            "CNN1D":   (seq_len, num_features),
            "MLP":     flat,
            "LSTM":    (seq_len, num_features),
            "GRU":     (seq_len, num_features),
            # Classical models receive flattened input
            "RF":      flat,
            "SVM":     flat,
            "XGBoost": flat,
            "CatBoost": flat,
        }
        if model_type:
            return {model_type: full[model_type]}
        # Default (no --nn): return only neural-network shapes
        return {k: full[k] for k in ("CNN1D", "MLP", "LSTM", "GRU")}

    @classmethod
    def get_feature_names(cls, scenario):
        return ["acc_x", "acc_y", "acc_z", "magacc", "gyr_x", "gyr_y", "gyr_z", "maggyr"]
    
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
        # For deterministic DataLoader shuffling
        cls.TORCH_GENERATOR = torch.Generator()
        cls.TORCH_GENERATOR.manual_seed(seed)
