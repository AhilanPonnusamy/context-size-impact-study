import os
import time
import pandas as pd
import requests
import random
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from bs4 import BeautifulSoup

# --- Configuration ---
VLLM_URL = "http://127.0.0.1:8000/v1"
# --- IMPORTANT: Change this to the model your vLLM server is running ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_ID = "Qwen/Qwen2-7B-Instruct" 
HAYSTACK_URL = "http://www.paulgraham.com/cities.html"
# --- UPDATED: New context sizes for this experiment ---
CONTEXT_SIZES_TO_TEST = [512, 1024, 2048, 4096]
# --- UPDATED: Reduced to 3 grand runs for efficiency ---
TOTAL_GRAND_RUNS = 3
# --- Use a new output file for this experiment ---
OUTPUT_CSV_PATH = "experiment_B_needle_haystack_results_llama3.csv"

# --- The "Needle" Configuration ---
NEEDLE = "The key to building truly great software is to remember that the best food in Melbourne is the parma at the local pub."
NEEDLE_QUESTION = "What is the key to building truly great software?"
EXPECTED_NEEDLE_RESPONSE = "melbourne" # A unique keyword to check for in the response

console = Console()

def get_haystack_text(url, tokenizer):
    """Downloads, extracts, and tokenizes the full text content from the URL."""
    console.print(f"--- Downloading and tokenizing 'haystack' from [cyan]{url}[/cyan] ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        tokens = tokenizer.encode(text)
        console.print(f"[green]Successfully prepared haystack with {len(tokens)} tokens.[/green]")
        return tokens
    except Exception as e:
        console.print(f"[bold red]Failed to download or process haystack: {e}[/bold red]")
        return None

def inject_needle(haystack_tokens, needle_tokens, position_ratio):
    """Injects the needle tokens into the haystack at a specific position ratio (0.0 to 1.0)."""
    insertion_point = int(len(haystack_tokens) * position_ratio)
    return haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]

def main():
    """Main function to orchestrate the 'Needle in a Haystack' experiment."""
    console.print(Panel(f"[bold cyan]Large Context Experiment: Part B - Needle in a Haystack ({MODEL_ID})[/bold cyan]", border_style="green"))

    # --- Setup ---
    console.print("--- Initializing tokenizer and vLLM client ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = ChatOpenAI(
        openai_api_base=VLLM_URL,
        api_key="NOT_USED",
        model=MODEL_ID,
        temperature=0,
    )
    
    full_haystack_tokens = get_haystack_text(HAYSTACK_URL, tokenizer)
    if not full_haystack_tokens:
        return
        
    needle_tokens = tokenizer.encode(NEEDLE)
    all_results = []
    
    # --- Outer loop for the 3 "Grand Runs" ---
    for grand_run_num in range(1, TOTAL_GRAND_RUNS + 1):
        console.print(Panel(f"Starting Grand Run {grand_run_num} of {TOTAL_GRAND_RUNS}", border_style="magenta", padding=(1,2)))
        
        test_configs = []
        for size in CONTEXT_SIZES_TO_TEST:
            for position_name, position_ratio in [("start", 0.05), ("middle", 0.50), ("end", 0.95)]:
                test_configs.append({"size": size, "position_name": position_name, "position_ratio": position_ratio})
        
        random.shuffle(test_configs)

        for config in track(test_configs, description=f"[Grand Run {grand_run_num}] Executing tests..."):
            context_size = config["size"]
            position_name = config["position_name"]
            position_ratio = config["position_ratio"]

            # Prepare the specific haystack for this test
            # Ensure there's space for the needle
            truncated_haystack = full_haystack_tokens[:context_size - len(needle_tokens)]
            haystack_with_needle_tokens = inject_needle(truncated_haystack, needle_tokens, position_ratio)
            haystack_with_needle_text = tokenizer.decode(haystack_with_needle_tokens)
            
            prompt = ChatPromptTemplate.from_messages([
                ("human", f"Please carefully read the following document and then answer the question at the end.\n\nDOCUMENT:\n{haystack_with_needle_text}\n\nQUESTION:\n{NEEDLE_QUESTION}")
            ])
            chain = prompt | llm

            start_time = time.time()
            response = chain.invoke({})
            end_time = time.time()
            
            response_time = end_time - start_time
            model_response_text = response.content
            accuracy = 1 if EXPECTED_NEEDLE_RESPONSE in model_response_text.lower() else 0
            
            all_results.append({
                "grand_run": grand_run_num,
                "model_id": MODEL_ID,
                "context_size": len(haystack_with_needle_tokens),
                "needle_position": position_name,
                "response_time": response_time,
                "accuracy": accuracy,
                "model_response": model_response_text
            })
            
        # --- NEW: Incremental saving after each grand run ---
        console.print(f"\n--- Grand Run {grand_run_num} complete. Saving intermediate results... ---")
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        console.print(f"--- Results saved to {OUTPUT_CSV_PATH} ---")

        if grand_run_num < TOTAL_GRAND_RUNS:
            console.print(f"\n[bold blue]Pausing for 30 seconds...[/bold blue]")
            time.sleep(30)

    # --- Final Analysis ---
    console.print("\n" + "="*50)
    console.print("[bold green]All experimental runs are complete. Finalizing results...[/bold green]")
    console.print("="*50 + "\n")
    
    summary = results_df.groupby(['context_size', 'needle_position']).agg(
        avg_response_time=('response_time', 'mean'),
        avg_accuracy=('accuracy', 'mean')
    ).reset_index()

    summary_table = Table(title=f"Needle in Haystack Summary for {MODEL_ID}", show_header=True, header_style="bold magenta")
    summary_table.add_column("Context Size (Tokens)", justify="right")
    summary_table.add_column("Needle Position")
    summary_table.add_column("Avg. Response Time (s)", justify="right")
    summary_table.add_column("Avg. Accuracy", justify="right")

    for _, row in summary.iterrows():
        summary_table.add_row(
            str(row['context_size']),
            row['needle_position'],
            f"{row['avg_response_time']:.4f}",
            f"{row['avg_accuracy']:.2%}"
        )
    
    console.print(summary_table)
    console.print(Panel("[bold magenta]Experiment B complete![/bold magenta]"))

if __name__ == "__main__":
    main()

