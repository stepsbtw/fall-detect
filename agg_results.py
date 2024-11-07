import os
import json

agg_results = {}

def define_key_subkey(filename):
    key = ""
    subkey = ""

    aux = filename.split("_")

    if filename.startswith("Sc1") or filename.startswith("Sc_3") or filename.startswith("Sc_4"):
        def define_key(x): return "_".join(x[:3] + x[4:])

    elif filename.startswith("Sc_2"):
        def define_key(x): return "_".join(x[:4] + x[5:])

    key = define_key(aux)
    subkey = "CNN1D" if filename.find(
        "CNN1D") > -1 else "MLP" if filename.find("MLP") > -1 else ""

    return key, subkey

# Para cada resultado '.json' registrado no diretório results, carregue o arquivo e agrupe-o em um novo json file
for i, filename in enumerate(sorted(os.listdir("results"))):
    if filename.endswith(".json"):
        with open(f"results/{filename}", "r") as file:

            key, subkey = define_key_subkey(filename[:-5])
            # Temos que aplicar essa lógica para não sobrescrever agg_results[key][subkey]
            if agg_results.get(key) is None:
                agg_results[key] = {}

            agg_results[key].update({subkey: json.load(file)})

with open("dump.json", "w") as f:
    json.dump(agg_results, f, indent=4)

print(f"{i+1} elementos agrupados em dump.json")
