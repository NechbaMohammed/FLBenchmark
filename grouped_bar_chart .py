import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# --- CONFIGURATION ---
BASE_EXPERIMENTS_DIR = os.path.join('results', 'experiments')
OUTPUT_DIR = os.path.join('results', 'plots', 'grouped_bar_chart')
# ---------------------

def find_data_files(base_path):
    """Finds all 'experiment_times_with_accuracy.csv' files within the specified base path."""
    data_files = []
    if not os.path.exists(base_path):
        print(f"Error: The specified base directory does not exist: '{base_path}'")
        return data_files

    for method_name in os.listdir(base_path):
        method_dir = os.path.join(base_path, method_name)
        if os.path.isdir(method_dir):
            file_path = os.path.join(method_dir, 'experiment_times_with_accuracy.csv')
            if os.path.exists(file_path):
                data_files.append((method_name, file_path))
    return data_files

def load_and_combine_data(data_files):
    """Loads all found CSVs into a single pandas DataFrame."""
    all_dfs = []
    for method_name, file_path in data_files:
        try:
            df = pd.read_csv(file_path)
            df['method'] = method_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Could not read or process {file_path}: {e}")
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def create_strategy_label(row):
    """Creates a user-friendly and descriptive label for the partitioning strategy."""
    partition = row['partitioning']
    try:
        params = ast.literal_eval(str(row['params']))
    except (ValueError, SyntaxError):
        params = {}

    if partition == 'iid':
        return 'IID\n(Uniform clients)'
    elif partition == 'dirichlet':
        alpha = params.get('alpha', 'N/A')
        return r'Dirichlet'+'\n'+r'($p_k \sim \mathrm{{Dir}}(0.5)$)'
    elif partition == 'label_quantity':
        labels = params.get('labels_per_client', 'N/A')
        if labels == 1:
            return '#C=1\n(Extreme label skew)'
        if labels == 2:
            return '#C=2\n(Moderate label skew)'
        if labels == 3:
            return '#C=3\n(High label skew)'
        return f'#C={labels}'
    elif partition == 'iid_noniid':
        return r'Quantity skew' + '\n' + r'($q \sim \mathrm{Dir}(0.5)$)'
    elif partition == 'noise':
        return r'Feature skew' + '\n' + r'($\hat{x} \sim Gau(0.1)$)'
    else:
        return partition.capitalize()

def generate_plot(df, dataset, metric, output_dir):
    """Generates and saves a single plot with hatching for a given dataset and metric."""
    plot_df = df[df['dataset'] == dataset].copy()
    plot_df = plot_df.groupby(['method', 'Strategy'])[metric].mean().reset_index()
    pivot_table = plot_df.pivot_table(index='Strategy', columns='method', values=metric)

    # UPDATED: The final strategy order for the x-axis
    strategy_order = [
        'IID\n(Uniform clients)',
        r'Quantity skew' + '\n' + r'($q \sim \mathrm{Dir}(0.5)$)',
        r'Feature skew' + '\n' + r'($\hat{x} \sim Gau(0.1)$)',
        r'Dirichlet'+'\n'+r'($p_k \sim \mathrm{{Dir}}(0.5)$)',
        '#C=3\n(High label skew)',
        '#C=2\n(Moderate label skew)',
        '#C=1\n(Extreme label skew)'
    ]
    
    ordered_strategies = [s for s in strategy_order if s in pivot_table.index]
    pivot_table = pivot_table.reindex(ordered_strategies)

    if pivot_table.empty:
        print(f"Warning: No data to plot for {dataset} and {metric}. Skipping.")
        return

    algorithms = pivot_table.columns.tolist()
    partition_strategies = pivot_table.index.tolist()

    x = np.arange(len(partition_strategies))
    num_algorithms = len(algorithms)
    total_width = 0.8
    bar_width = total_width / num_algorithms

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', '..', '**']
    colors = plt.cm.get_cmap('viridis', num_algorithms)
    algorithms.remove('depthfl')
    for i, algo in enumerate(algorithms):
        offset = (i - num_algorithms / 2 + 0.5) * bar_width
        values = pivot_table[algo].fillna(0)
        print(algo)
        print(values)
        ax.bar(x + offset, values, width=bar_width, label=algo,
               hatch=hatches[i % len(hatches)], color=colors(i),
               edgecolor='black', alpha=0.9)

    y_label = 'Execution Time (seconds)'
    title = " "

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Data Partitioning Strategy', fontsize=12)
    ax.set_xticks(x, partition_strategies, fontsize=10)

    ax.legend(title="Algorithms", bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()

    output_filename = os.path.join(output_dir, f"plot_{dataset}_time.pdf")
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"--> Plot saved successfully to {output_filename}")
    plt.close(fig)

# --- Main script execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved in: '{OUTPUT_DIR}'")

    files_to_process = find_data_files(BASE_EXPERIMENTS_DIR)

    if not files_to_process:
        print("Error: No 'experiment_times_with_accuracy.csv' files found.")
    else:
        print("Found data files for methods:", [f[0] for f in files_to_process])
        master_df = load_and_combine_data(files_to_process)

        if not master_df.empty:
            #master_df['time_hours'] = master_df['time_sec'] / 3600
            master_df['Strategy'] = master_df.apply(create_strategy_label, axis=1)

            datasets = master_df['dataset'].unique()
            metric_to_plot = 'time_sec'
            
            for ds in datasets:
                print(f"\nGenerating plot for Dataset: '{ds}', Metric: '{metric_to_plot}'...")
                generate_plot(master_df, ds, metric_to_plot, OUTPUT_DIR)
            
            print("\n✅ All plots have been generated.")
        else:
            print("Error: Failed to load any data.")