import os
import csv
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def extract_data_from_csv(csv_file):
    """Extract rounds and accuracies from aggregated CSV file"""
    rounds = []
    accuracies = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            rounds.append(int(row[0]))
            accuracies.append(float(row[1]))
    return rounds, accuracies

def plot_federated_learning_curves(all_methods_data, rounds_num, epochs_num, output_path=None):
    """Plot learning curves for multiple methods with rounds and epochs in title"""
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Define styles for each method
    method_styles = {
        'fedavg': {'label': 'FedAvg', 'color': '#1f77b4', 'linestyle': '-', 'marker': '^'},
        'fedprox': {'label': 'FedProx, μ=0.1', 'color': '#ff7f0e', 'linestyle': '--', 'marker': '*'},
        'fedadam': {'label': 'Adam', 'color': '#2ca02c', 'linestyle': ':', 'marker': 'o'},
        'fedadagrad': {'label': 'FedAdagrad', 'color': '#d62728', 'linestyle': '-.', 'marker': 's'},
        'fedyogi': {'label': 'FedYogi', 'color': '#9467bd', 'linestyle': (0, (3, 1, 1, 1)), 'marker': 'D'},
        'fednova': {'label': 'FedNova', 'color': '#8c564b', 'linestyle': (0, (5, 1)), 'marker': 'P'},
        'scaffold': {'label': 'Scaffold', 'color': '#e377c2', 'linestyle': (0, (1, 1)), 'marker': 'X'},
        'moon': {'label': 'Moon', 'color': '#7f7f7f', 'linestyle': (0, (3, 5, 1, 5)), 'marker': 'H'},
        'fedbn': {'label': 'FedBN', 'color': '#17becf', 'linestyle': (0, (5, 5)), 'marker': '>'}

    }

    # Plot each method's data
    for method, (rounds, accuracies) in all_methods_data.items():
        if method.lower() in method_styles:
            style = method_styles[method.lower()]
            plt.plot(rounds, accuracies, 
                    label=style['label'], 
                    color=style['color'], 
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=6)

    # Formatting
    plt.xlabel('Communication round', fontsize=20)
    plt.ylabel('Test accuracy', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    #plt.title(f'Rounds: {rounds_num}, Epochs: {epochs_num}', fontsize=16)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        plt.show()

def extract_rounds_epochs(filename):
    """Extract rounds and epochs numbers from filename"""
    match = re.search(r'_rounds(\d+)_epochs(\d+)\.csv$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    # Configuration
    agg_dir = "results/agg_experiments"
    plots_dir = "results/plots/training_Curves"
    os.makedirs(plots_dir, exist_ok=True)

    # Find all aggregated CSV files
    csv_files = glob.glob(os.path.join(agg_dir, "**", "*.csv"), recursive=True)

    # Organize files by experiment configuration and rounds/epochs
    experiment_groups = defaultdict(lambda: defaultdict(list))
    
    for csv_file in csv_files:
        parts = csv_file.split(os.sep)
        filename = os.path.basename(csv_file)
        
        # Extract method, rounds, and epochs from filename
        method = filename.split('_')[0]
        rounds_num, epochs_num = extract_rounds_epochs(filename)
        
        if rounds_num is None or epochs_num is None:
            continue  # Skip files that don't match our pattern
            
        experiment_key = os.path.join(*parts[1:-1])  # Skip agg_dir and filename
        config_key = (rounds_num, epochs_num)
        
        experiment_groups[experiment_key][config_key].append((method, csv_file))

    # Process each experiment group and each rounds/epochs configuration
    for experiment_key, configs in experiment_groups.items():
        for (rounds_num, epochs_num), method_files in configs.items():
            all_methods_data = {}
            
            # Collect data for all methods in this experiment config
            for method, csv_file in method_files:
                rounds, accuracies = extract_data_from_csv(csv_file)
                all_methods_data[method] = (rounds, accuracies)

            # Only plot if we have at least 2 methods to compare
            if len(all_methods_data) >= 2:
                # Create output path with rounds and epochs in filename
                dataset = os.path.basename(experiment_key)
                partitioning = os.path.dirname(experiment_key)
                plot_filename = f"{dataset}_rounds{rounds_num}_epochs{epochs_num}_training_curve.pdf"
                output_path = os.path.join(plots_dir, partitioning, plot_filename)

                # Generate and save plot
                plot_federated_learning_curves(all_methods_data, rounds_num, epochs_num, output_path)
                print(f"Saved plot: {output_path}")

if __name__ == "__main__":
    main()