import os
import subprocess
import time
from collections import deque
from typing import List

MAX_PROCESSES_AT_ONCE = 1

rounds =50 # Number of rounds
epochs = "10 20 40" # List Number of local epochs
runs = 3 # Number of runs to execute
num_clients = 10 # for all partitioning except cifar100 
alpha = 0.5 # for dirichlet partitioning
segma = 0.1 # for noise partitioning
similarity = 0.5 # for iid_noniid partitioning
labels_per_client = "1 2 3" # for label_quantity partitioning
mu = 0.01 # for fedprox
beta_1=0.9 # for fedadam and fedyogi
beta_2=0.99 # for fedadam and fedyogi
learning_rate = 0.01 # for fednova and scaffold


# Define the methods and datasets to run
methods =["fedavg","fedprox","fedadagrad","fedadam","fedyogi","fednova","scaffold","moon","fedbn"] 
datasets = ["mnist","fmnist", "cifar10","cifar100","svhn","cinic10","fedisic2019","adult", "fcube","femnist"]
partitioning ={"mnist": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
               "fmnist": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
                "svhn": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
                "cinic10": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
                "fedisic2019": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
            "cifar10": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
            "cifar100": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
            "adult": ["label_quantity", "dirichlet", "iid_noniid", "iid"],
            "fcube": ["synthetic"],
            "femnist": ["real-world"]
            }
 

# Boucle pour exécuter les commandes
commands: deque = deque()

for method in methods:
    for dataset in datasets:
        if dataset == "cifar100":
            num_clients = 100
        else:
            num_clients = 10
        for part in partitioning[dataset]:
            if part == "label_quantity":
                if dataset == "adult":
                    cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --labels_per_client {1} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
                else:
                    cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --labels_per_client {labels_per_client} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "dirichlet":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --alpha {alpha} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "iid_noniid":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --similarity {similarity} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "noise":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --segma {segma} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "iid":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --num_clients {num_clients} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "synthetic":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --num_clients {4} --rounds {rounds} --epochs {epochs} --runs {runs}"
            elif part == "real-world":
                cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --num_clients {100} --rounds {rounds} --epochs {epochs} --runs {runs}"
            else:
                continue  # Ignorer les partitions non reconnues
            if method=="fedprox":
                cmd = cmd + f" --mu {mu}"
            elif method=="fedadam" or method=="fedyogi":
                cmd = cmd + f" --beta_1 {beta_1} --beta_2 {beta_2}"
            # add learning rate
            cmd = cmd + f" --learning_rate {learning_rate}"
            commands.append(cmd)



# run max_processes_at_once processes at once with 10 second sleep interval
# in between those processes until all commands are done
processes: List = []
while len(commands) > 0:
    while len(processes) < MAX_PROCESSES_AT_ONCE and len(commands) > 0:
        cmd = commands.popleft()
        print(cmd)
        processes.append(subprocess.Popen(cmd, shell=True))
        # sleep for 10 seconds to give the process time to start
        time.sleep(10)
    for p in processes:
        if p.poll() is not None:
            processes.remove(p)