import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
import os

# --- Configuration ---
# Update this dictionary with the model names for the legend
# and the corresponding CSV file paths.
RESULT_FILES = {
    "Meta-Llama-3.1-8B": "experiment_B_needle_haystack_results_llama3.csv",
    "Qwen2-7B": "experiment_B_needle_haystack_results_Qwen.csv",
}

# Define the output filenames for the generated charts
OUTPUT_PERFORMANCE_CHART = "needle_haystack_performance_grouped.png"
OUTPUT_ACCURACY_CHART = "needle_haystack_accuracy_grouped.png"

console = Console()

def load_and_merge_data():
    """Loads all result CSVs, assigns model names, and merges them."""
    all_dfs = []
    console.print("--- Loading and merging result files ---")
    for model_name, file_path in RESULT_FILES.items():
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error: Result file not found: '{file_path}'[/bold red]")
            console.print(f"[yellow]Skipping analysis for {model_name}. Please ensure the file exists.[/yellow]")
            continue
        
        try:
            df = pd.read_csv(file_path)
            # Standardize the model name from the 'model_id' field for the legend
            df['model_name'] = model_name
            all_dfs.append(df)
            console.print(f"  [green]Loaded {file_path} for {model_name}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading {file_path}: {e}[/bold red]")
    
    if not all_dfs:
        return None
        
    return pd.concat(all_dfs, ignore_index=True)

def generate_graphs(df: pd.DataFrame):
    """Generates comparative grouped bar charts for performance and accuracy."""
    console.print("\n--- Generating comparative analysis graphs ---")
    
    # Use a professional plot style
    sns.set_theme(style="whitegrid")

    # --- Enforce a logical order for sorting and plotting ---
    category_order = ['start', 'middle', 'end']
    df['needle_position'] = pd.Categorical(df['needle_position'], categories=category_order, ordered=True)
    df.sort_values(by=['needle_position', 'context_size'], inplace=True)
    
    # Create a combined x-axis label for clarity
    df['x_label'] = df['needle_position'].astype(str) + " @ " + df['context_size'].astype(str)

    # --- 1. Performance Grouped Bar Chart ---
    console.print("  - Generating Performance Chart...")
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    sns.barplot(data=df, x='x_label', y='response_time', hue='model_name', ax=ax1, palette='viridis')
    
    ax1.set_title('Performance by Needle Position and Context Size', fontsize=18, pad=20)
    ax1.set_xlabel('Test Case (Position @ Context Tokens)', fontsize=12)
    ax1.set_ylabel('Average Response Time (seconds)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PERFORMANCE_CHART)
    console.print(f"  [green]Performance graph saved to [cyan]{OUTPUT_PERFORMANCE_CHART}[/cyan][/green]")

    # --- 2. Accuracy Grouped Bar Chart ---
    console.print("  - Generating Accuracy Chart...")
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    sns.barplot(data=df, x='x_label', y='accuracy', hue='model_name', ax=ax2, palette='plasma')

    ax2.set_title('Accuracy by Needle Position and Context Size', fontsize=18, pad=20)
    ax2.set_xlabel('Test Case (Position @ Context Tokens)', fontsize=12)
    ax2.set_ylabel('Average Accuracy', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.legend(title='Model')
    ax2.set_ylim(0, 1.05) 
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    plt.tight_layout()
    plt.savefig(OUTPUT_ACCURACY_CHART)
    console.print(f"  [green]Accuracy graph saved to [cyan]{OUTPUT_ACCURACY_CHART}[/cyan][/green]")


def main():
    """Main function to run the comparative analysis."""
    console.print(Panel("[bold cyan]Comparative Analysis by Needle Position & Context Size[/bold cyan]", border_style="green"))
    
    merged_df = load_and_merge_data()
    
    if merged_df is None or merged_df.empty:
        console.print("[bold red]No data to analyze. Exiting.[/bold red]")
        return

    # --- Print Detailed Summary Table ---
    category_order = ['start', 'middle', 'end']
    merged_df['needle_position'] = pd.Categorical(merged_df['needle_position'], categories=category_order, ordered=True)

    # Group by all three columns for a more detailed summary
    summary = merged_df.groupby(['model_name', 'needle_position', 'context_size'], observed=False).agg(
        avg_response_time=('response_time', 'mean'),
        avg_accuracy=('accuracy', 'mean')
    ).reset_index()
    
    console.print("\n--- Overall Performance and Accuracy Summary ---")
    
    try:
        # Pivot for a more readable console output
        pivot_summary = summary.pivot_table(
            index=['needle_position', 'context_size'], 
            columns='model_name', 
            values=['avg_response_time', 'avg_accuracy']
        )
        console.print(pivot_summary.round(4))
    except Exception as e:
        console.print(f"[yellow]Could not create pivot table: {e}. Displaying raw summary instead.[/yellow]")
        console.print(summary.round(4).to_string())

    # --- Generate Graphs ---
    generate_graphs(merged_df.copy())
    
    console.print(Panel("[bold magenta]Comparative analysis complete![/bold magenta]"))

if __name__ == "__main__":
    main()

