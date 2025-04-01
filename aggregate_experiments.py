import os
import csv
import glob
import math
import re
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Aggregate federated learning results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Base directory containing the results")
    parser.add_argument("--agg_dir", type=str, default="results/agg_experiments",
                       help="Directory to store aggregated results")
    args = parser.parse_args()

    # Create the aggregation directory if it doesn't exist
    os.makedirs(args.agg_dir, exist_ok=True)

    # Find all CSV files in the results directory
    csv_files = glob.glob(os.path.join(args.results_dir, "**", "*.csv"), recursive=True)

    # Organize files by partitioning, dataset, and method
    organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in csv_files:
        parts = file_path.split(os.sep)
        
        # Expected structure: method/dataset/partitioning/.../metrics/file.csv
        if len(parts) < 6:
            continue  # Skip invalid paths

        method = parts[2]
        dataset = parts[3]
        partitioning_parts = parts[4:-2]  # Get all partitioning components
        
        # Construct partitioning key
        partitioning_key = "/".join(partitioning_parts)
        
        # Extract parameters from filename using regex
        filename = parts[-1]
        rounds = re.search(r'rounds(\d+)', filename).group(1)
        epochs = re.search(r'epochs(\d+)', filename).group(1)
        
        # Store in organized structure
        organized_files[partitioning_key][dataset][method].append({
            "path": file_path,
            "rounds": rounds,
            "epochs": epochs
        })

    # Process each group
    for partitioning, datasets in organized_files.items():
        for dataset, methods in datasets.items():
            for method, files in methods.items():
                # Group by rounds and epochs configuration
                config_groups = defaultdict(list)
                for file_info in files:
                    key = f"rounds{file_info['rounds']}_epochs{file_info['epochs']}"
                    config_groups[key].append(file_info["path"])

                # Process each configuration group
                for config, file_paths in config_groups.items():
                    all_accuracies = []
                    
                    # Read all files
                    for fp in file_paths:
                        with open(fp, 'r') as f:
                            reader = csv.reader(f)
                            next(reader)  # Skip header
                            accuracies = [float(row[1]) for row in reader]
                            all_accuracies.append(accuracies)
                    
                    # Handle varying rounds
                    min_rounds = min(len(acc) for acc in all_accuracies)
                    all_accuracies = [acc[:min_rounds] for acc in all_accuracies]

                    # Calculate averages and std
                    aggregated = []
                    for i in range(min_rounds):
                        values = [acc[i] for acc in all_accuracies]
                        avg = sum(values) / len(values)
                        std = math.sqrt(sum((x-avg)**2 for x in values)/len(values))
                        aggregated.append((i, avg, std))

                    # Create output directory
                    output_dir = os.path.join(
                        args.agg_dir,
                        partitioning,
                        dataset
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    # Save aggregated results
                    output_file = os.path.join(output_dir, f"{method}_{config}.csv")
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Round', 'Avg Accuracy', 'Std Accuracy'])
                        for row in aggregated:
                            writer.writerow([
                                row[0],
                                f"{row[1]:.4f}",
                                f"{row[2]:.4f}"
                            ])
                    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()