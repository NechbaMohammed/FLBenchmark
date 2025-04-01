import os
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def extract_data_from_csv(csv_file):
    """Extract final accuracy from CSV file"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        # Get the last row (final accuracy)
        final_row = None
        for row in reader:
            final_row = row
        return float(final_row[1]) if final_row else None

def extract_rounds_epochs(filename):
    """Extract rounds and epochs numbers from filename"""
    match = re.search(r'_rounds(\d+)_epochs(\d+)\.csv$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def plot_local_epoch_comparison(epochs_data, output_path=None):
    """Plot bar chart comparing methods across different local epochs"""
    # Get unique epochs and methods
    local_epochs = sorted(list({epoch for method, epoch, acc in epochs_data}))
    methods = sorted(list({method for method, epoch, acc in epochs_data}))
    
    # Organize data by method
    method_accuracies = defaultdict(dict)
    for method, epoch, acc in epochs_data:
        method_accuracies[method][epoch] = acc
    
    # Define styles for each method
    method_styles = {
        'fedavg': {'label': 'FedAvg', 'color': 'tab:blue', 'hatch': '//'},
        'fedprox': {'label': 'FedProx, μ=0.01', 'color': 'tab:orange', 'hatch': '\\\\'},
        'scaffold': {'label': 'SCAFFOLD', 'color': 'tab:green', 'hatch': ''},
        'fednova': {'label': 'FedNova', 'color': 'tab:red', 'hatch': '--'},
        'fedyogi': {'label': 'FedYogi', 'color': '#fdc086', 'hatch': ''},
        'fedadam': {'label': 'FedAdam', 'color': '#ffff99', 'hatch': 'xx'},
        'fedadagrad': {'label': 'FedAdagrad', 'color': 'tab:pink', 'hatch': 'oo'},
        'moon': {'label': 'MOON', 'color': '#17becf', 'hatch': '*'},
        'fedbn': {'label': 'FedBN', 'color': 'tab:olive', 'hatch': '..'}
    }
    
    # Set up plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.08
    r = np.arange(len(local_epochs))
    
    # Plot each method
    for i, method in enumerate(methods):
        if method.lower() in method_styles:
            style = method_styles[method.lower()]
            accuracies = [method_accuracies[method].get(epoch, 0) for epoch in local_epochs]
            plt.bar(r + i*bar_width, accuracies, color=style['color'], 
                   width=bar_width, label=style['label'], hatch=style['hatch'])
    
    # Formatting
    plt.xlabel('Local epochs', fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14)
    plt.xticks([r + bar_width*(len(methods)/2 - 0.5) for r in range(len(local_epochs))], 
               local_epochs, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Configuration
    agg_dir = "results/agg_experiments"
    plots_dir = "results/plots/local_epochs"
    os.makedirs(plots_dir, exist_ok=True)

    # Find all aggregated CSV files
    csv_files = glob.glob(os.path.join(agg_dir, "**", "*.csv"), recursive=True)

    # Organize data by experiment configuration and method
    experiment_groups = defaultdict(list)
    
    for csv_file in csv_files:
        parts = csv_file.split(os.sep)
        filename = os.path.basename(csv_file)
        
        # Extract method, rounds, and epochs from filename
        method = filename.split('_')[0]
        rounds_num, epochs_num = extract_rounds_epochs(filename)
        
        if rounds_num is None or epochs_num is None:
            continue
            
        # Get final accuracy
        final_acc = extract_data_from_csv(csv_file)
        if final_acc is None:
            continue
            
        experiment_key = os.path.join(*parts[1:-1])  # Skip agg_dir and filename
        experiment_groups[experiment_key].append((method.lower(), epochs_num, final_acc))

    # Process each experiment group
    for experiment_key, epochs_data in experiment_groups.items():
        # Only plot if we have data for multiple epochs
        unique_epochs = {epoch for _, epoch, _ in epochs_data}
        if len(unique_epochs) > 1:
            dataset = os.path.basename(experiment_key)
            partitioning = os.path.dirname(experiment_key)
            plot_filename = f"{dataset}_local_epochs_comparison.pdf"
            output_path = os.path.join(plots_dir, partitioning, plot_filename)
            
            plot_local_epoch_comparison(epochs_data, output_path)
            print(f"Saved plot: {output_path}")

if __name__ == "__main__":
    main()