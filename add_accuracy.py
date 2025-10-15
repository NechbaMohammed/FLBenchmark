import os
import pandas as pd
import ast # Used to safely evaluate the string representation of a dictionary

def find_accuracy_from_experiment(row):
    """
    Constructs the path to an experiment's results file, reads it,
    and returns the final accuracy.
    """
    base_path = 'results\experiments'
    
    # --- 1. Construct the path from the row's data ---
    method = row['method']
    dataset = row['dataset']
    partitioning = row['partitioning']
    run = row['run']
    
    # Safely convert the params string to a dictionary
    try:
        params = ast.literal_eval(row['params'])
    except (ValueError, SyntaxError):
        params = {}

    # Build the path based on the partitioning type
    # This logic maps the 'partitioning' name to the directory structure
    sub_path = ''
    if partitioning == 'label_quantity':
        labels_per_client = params.get('labels_per_client')
        if labels_per_client:
            # e.g., experiments/dasha/mnist/label_quantity/C1/metrics
            sub_path = os.path.join('label_distribution','label_quantity', f'C{labels_per_client}', 'metrics')
    
    elif partitioning == 'dirichlet':
        alpha = params.get('alpha')
        if alpha is not None:
            # e.g., experiments/dasha/fmnist/label_distribution/dirichlet/alpha0.5/metrics
            sub_path = os.path.join('label_distribution', 'dirichlet', f'alpha{alpha}', 'metrics')
    
    elif partitioning == 'iid':
        # Assuming 'iid' maps to 'homogeneous_partition' based on common FL terms
        sub_path = os.path.join('homogeneous_partition', 'metrics')
    elif partitioning == 'noise':
        # Assuming 'noise' maps to 'homogeneous_partition' based on common FL terms
        sub_path = os.path.join('feature_distribution', 'metrics')
    else:
        # Default fallback for other types like 'noise', 'iid_noniid'
        # This assumes a direct mapping: experiments/dasha/mnist/noise/metrics
        sub_path = os.path.join('quantity_skew', 'metrics')

    # The full path to the metrics directory
    metrics_dir = os.path.join(base_path, method, dataset, sub_path)

    # --- 2. Find the correct CSV file in the metrics directory ---
    if not os.path.exists(metrics_dir):
        print(f"Warning: Directory not found -> {metrics_dir}")
        return None

    target_file = None
    run_identifier = f"run{run}.csv"
    for filename in os.listdir(metrics_dir):
        # Find the file for the correct run
        if filename.endswith(run_identifier):
            target_file = os.path.join(metrics_dir, filename)
            break
    
    if not target_file:
        print(f"Warning: Could not find results file for run {run} in -> {metrics_dir}")
        return None

    # --- 3. Read the file and extract the last accuracy value ---
    try:
        results_df = pd.read_csv(target_file)
        
        # Define possible names for the accuracy column
        accuracy_column_names = ['Accuracy']
        
        # Find the correct accuracy column in the dataframe
        accuracy_col = next((col for col in accuracy_column_names if col in results_df.columns), None)

        if accuracy_col:
            # Get the value from the last row of the accuracy column
            last_accuracy = results_df[accuracy_col].iloc[-1]
            return last_accuracy
        else:
            print(f"Warning: No accuracy column found in -> {target_file}")
            return None
            
    except Exception as e:
        print(f"Error reading file {target_file}: {e}")
        return None


# --- Main script execution ---

# Define input and output filenames
methods = ['dasha','fedexp', 'depthfl', 'ditto','fedavg','fedrep']
for method in methods:
    input_csv = f'results/experiments/{method}/experiment_times.csv'
    output_csv = f'results/experiments/{method}/experiment_times_with_accuracy.csv'

    # Check if the input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
    else:
        # Load the experiment times data
        df = pd.read_csv(input_csv)

        print("Processing experiments to find final accuracy...")
        
        # Apply the function to each row to get the accuracies
        # The result will be a new Series (like a list) of accuracy values
        df['Accuracy'] = df.apply(find_accuracy_from_experiment, axis=1)

        # Save the updated dataframe to a new file
        df.to_csv(output_csv, index=False)

        print(f"\nProcessing complete. Data with accuracy column saved to '{output_csv}'.")