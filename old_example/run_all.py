# run_all_parallel.py

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# PARAMETERS
scenarios = [
    "Sc1_acc_T", "Sc1_gyr_T",
    "Sc_2_acc_T", "Sc_2_gyr_T",
    "Sc_3_T", "Sc_4_T"
]
positions = ["left", "chest", "right"]
networks = ["MLP", "CNN1D"]
label_type = "binary_one"

MAX_WORKERS = 4  # Adjust this depending on your CPU/RAM capacity

# Clean or create the results/ directory
if os.path.exists("results"):
    for file in os.listdir("results"):
        os.remove(os.path.join("results", file))
else:
    os.makedirs("results")

def run_training(scenario, position, network):
    print(f"🟡 Starting: {scenario}, {position}, {network}")
    command = [
        "python", "run_of_the_neural_network_model.py",
        "--scenario", scenario,
        "--position", position,
        "--label_type", label_type,
        "--neural_network_type", network
    ]
    try:
        subprocess.run(command, check=True)
        print(f"✅ Finished: {scenario}, {position}, {network}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {scenario}, {position}, {network}: {e}")

# Run all jobs in parallel
futures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for scenario in scenarios:
        for position in positions:
            for network in networks:
                futures.append(executor.submit(run_training, scenario, position, network))

    for future in as_completed(futures):
        pass  # All output is handled inside `run_training`

# Extract reports
print("\n📦 Extracting report.json files...")
subprocess.run(["bash", "extract_reports.sh"], check=True)

# Aggregate
print("\n📊 Aggregating results...")
subprocess.run(["python", "agg_results.py"], check=True)

print("\n🎉 All tasks completed.")
