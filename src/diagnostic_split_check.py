import numpy as np
import os
from collections import Counter

# Caminho do split salvo
base_out = os.path.join('output', 'CNN1D', 'chest', 'Sc_4_T', 'binary_two')
test_data_file = os.path.join(base_out, 'test_data.npz')

# Carregar splits
if not os.path.exists(test_data_file):
    raise FileNotFoundError(f'Arquivo não encontrado: {test_data_file}')
data = np.load(test_data_file)
X_trainval, y_trainval = data['X_trainval'], data['y_trainval']
X_test, y_test = data['X_test'], data['y_test']

# Split train/val igual ao final_training.py
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.2,  # igual ao Config.DATA_SPLIT['test_size']
    random_state=42,
    shuffle=False  # Para diagnóstico temporal
)

def class_dist(y):
    c = Counter(y if y.ndim == 1 else np.argmax(y, axis=1))
    total = sum(c.values())
    return {k: f'{v} ({v/total:.2%})' for k, v in c.items()}

# Distribuição das classes
train_dist = class_dist(y_train)
val_dist = class_dist(y_val)
test_dist = class_dist(y_test)

# Checar duplicidade entre splits
train_flat = X_train.reshape((X_train.shape[0], -1))
val_flat = X_val.reshape((X_val.shape[0], -1))
test_flat = X_test.reshape((X_test.shape[0], -1))

def count_overlap(a, b):
    a_set = set(map(bytes, a))
    b_set = set(map(bytes, b))
    return len(a_set & b_set)

overlap_train_val = count_overlap(train_flat, val_flat)
overlap_train_test = count_overlap(train_flat, test_flat)
overlap_val_test = count_overlap(val_flat, test_flat)

# Relatório
report = []
report.append('Distribuição das classes:')
report.append(f'Treino: {train_dist}')
report.append(f'Validação: {val_dist}')
report.append(f'Teste: {test_dist}')
report.append('\nOverlap de amostras idênticas:')
report.append(f'Treino x Validação: {overlap_train_val}')
report.append(f'Treino x Teste: {overlap_train_test}')
report.append(f'Validação x Teste: {overlap_val_test}')

report_txt = '\n'.join(report)
print(report_txt)

with open(os.path.join(base_out, 'diagnostic_split_report.txt'), 'w') as f:
    f.write(report_txt) 