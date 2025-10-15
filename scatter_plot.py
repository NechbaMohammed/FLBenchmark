import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import re

# --- CONFIGURATION ---
BASE_EXPERIMENTS_DIR = os.path.join('results', 'experiments')
OUTPUT_DIR = os.path.join('results', 'plots', 'scatter_plots')
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

def get_plot_title(row):
    """Creates a clean, single-line title for the plot from a row of data."""
    partition = row['partitioning']
    try:
        params = ast.literal_eval(str(row['params']))
    except (ValueError, SyntaxError):
        params = {}

    if partition == 'iid':
        return 'IID (Uniform clients)'
    elif partition == 'dirichlet':
        alpha = params.get('alpha', 'N/A')
        return f'Dirichlet Skew (alpha={alpha})'
    elif partition == 'label_quantity':
        labels = params.get('labels_per_client', 'N/A')
        if labels == 1: return 'Extreme Label Skew (#C=1)'
        if labels == 2: return 'Moderate Label Skew (#C=2)'
        if labels == 3: return 'High Label Skew (#C=3)'
        return f'Label Skew (#C={labels})'
    elif partition == 'iid_noniid':
        return 'Quantity Skew (Dir(0.5))'
    elif partition == 'noise':
        return 'Feature Skew (Gau(0.1))'
    else:
        return partition.capitalize()

def generate_scatter_plot(df_subset, dataset, partition_params, output_dir):
    """Generates and saves a publication-quality scatter plot for a specific experimental setup."""
    import matplotlib as mpl

    # Aggregate mean results
    # i need to remove depthfl  form df_subset
    df_subset = df_subset[df_subset['method'] != 'depthfl']
    plot_data = df_subset.groupby('method')[['time_sec', 'Accuracy']].mean().reset_index()
    if plot_data.empty:
        print(f"Warning: No data to plot for {dataset} - {partition_params}. Skipping.")
        return

    algorithms = plot_data['method'].tolist()
    execution_times = plot_data['time_sec'].tolist()
    accuracies = plot_data['Accuracy'].tolist()

    # --- Aesthetics ---
    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cmap = plt.cm.get_cmap('tab10', len(algorithms))
    ax.scatter(
        execution_times, accuracies,
        s=120, c=np.arange(len(algorithms)), cmap=cmap,
        edgecolors='black', linewidths=0.8, alpha=0.9
    )

    # Annotate each point with algorithm name
    for i, algo in enumerate(algorithms):
        ax.annotate(
            algo, (execution_times[i], accuracies[i]),
            textcoords="offset points", xytext=(6, 4),
            ha='left', fontsize=11, color='black'
        )

    # Smart axis limits
    ax.set_xscale('log')  # comment out if you prefer linear
    ymin = plot_data['Accuracy'].min() - 0.02
    ymax = plot_data['Accuracy'].max() + 0.02
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Total Execution Time (log seconds)', labelpad=8)
    ax.set_ylabel('Test Accuracy', labelpad=8)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_title(" ", pad=12, weight='bold')
    fig.tight_layout()

    # --- Save to file ---
    safe_partition_name = re.sub(r'[^a-zA-Z0-9_-]', '', f"{df_subset.iloc[0]['partitioning']}_{df_subset.iloc[0]['params']}").replace('__', '_')
    output_filename = os.path.join(output_dir, f"scatter_{dataset}_{safe_partition_name}.pdf")

    fig.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"--> Scatter plot saved to {output_filename}")
    plt.close(fig)


# --- Main script execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Scatter plots will be saved in: '{OUTPUT_DIR}'")

    files_to_process = find_data_files(BASE_EXPERIMENTS_DIR)

    if not files_to_process:
        print("Error: No 'experiment_times_with_accuracy.csv' files found.")
    else:
        print("Found data files for methods:", [f[0] for f in files_to_process])
        master_df = load_and_combine_data(files_to_process)

        if not master_df.empty:
            # Group data by each unique experimental setup
            # NOTE: We are NOT converting time to hours here
            grouped = master_df.groupby(['dataset', 'partitioning', 'params'])
            
            for (dataset, partition, params), subset_df in grouped:
                partition_id_for_log = f"partition={partition}, params={params}"
                print(f"\nGenerating scatter plot for: Dataset='{dataset}', Config='{partition_id_for_log}'")
                generate_scatter_plot(subset_df, dataset, partition_id_for_log, OUTPUT_DIR)
            
            print("\n✅ All scatter plots have been generated.")
        else:
            print("Error: Failed to load any data.")