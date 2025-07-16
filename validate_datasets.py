import os
import numpy as np
from config import Config

positions = ["left", "chest", "right"]
scenarios = list(Config.SCENARIOS.keys())

def main():
    print("Validação dos datasets gerados pelo generate_datasets:")
    for position in positions:
        data_path = os.path.join(Config.DATA_PATH, position)
        print(f"\n--- Posição: {position} ---")
        for scenario in scenarios:
            filename, expected_shape = Config.SCENARIOS[scenario]
            file_path = os.path.join(data_path, filename)
            if not os.path.exists(file_path):
                print(f"[MISSING] {scenario}: {file_path} não encontrado.")
                continue
            try:
                X = np.load(file_path)
                print(f"{scenario}: {filename} -> shape real: {X.shape}, shape esperado: {expected_shape}")
            except Exception as e:
                print(f"[ERROR] {scenario}: {filename} -> erro ao carregar: {e}")

if __name__ == "__main__":
    main() 