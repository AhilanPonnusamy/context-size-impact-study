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
#MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_ID = "Qwen/Qwen2-7B-Instruct"
MODEL_ID = "Mistralai/Mistral-7B-Instruct-v0.3"
HAYSTACK_URL = "http://www.paulgraham.com/cities.html"
CONTEXT_SIZES_TO_TEST = [1000, 2000, 4000]
# --- NEW: Configuration for the overnight run ---
TOTAL_GRAND_RUNS = 5 # The number of times to repeat the entire experiment
TOTAL_SUB_RUNS = 3   # The number of times to repeat the 25 questions for each setting
#OUTPUT_CSV_PATH = "experiment_A_overnight_results_llama31.csv"
OUTPUT_CSV_PATH = "experiment_A_overnight_results_Mistral0.3.csv"

console = Console()

# --- The 25 General Knowledge Questions ---
GENERAL_KNOWLEDGE_QUESTIONS = [
    {"question": "What is the capital of France?", "answer": "paris"},
    {"question": "Who wrote 'Hamlet'?", "answer": "shakespeare"},
    {"question": "What is the chemical symbol for water?", "answer": "h2o"},
    {"question": "In which year did the Titanic sink?", "answer": "1912"},
    {"question": "What planet is known as the Red Planet?", "answer": "mars"},
    {"question": "Who painted the Mona Lisa?", "answer": "vinci"},
    {"question": "What is the tallest mountain in the world?", "answer": "everest"},
    {"question": "What is the main ingredient in guacamole?", "answer": "avocado"},
    {"question": "How many continents are there?", "answer": "seven"},
    {"question": "Who was the first person to walk on the moon?", "answer": "armstrong"},
    {"question": "What is the currency of Japan?", "answer": "yen"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "diamond"},
    {"question": "Which ocean is the largest?", "answer": "pacific"},
    {"question": "Who invented the telephone?", "answer": "bell"},
    {"question": "What is the square root of 64?", "answer": "8"},
    {"question": "Which country is famous for its pyramids?", "answer": "egypt"},
    {"question": "What is the primary language spoken in Brazil?", "answer": "portuguese"},
    {"question": "Who discovered penicillin?", "answer": "fleming"},
    {"question": "What is the boiling point of water at sea level?", "answer": "100"},
    {"question": "Which artist cut off his own ear?", "answer": "van gogh"},
    {"question": "What is the largest animal in the world?", "answer": "blue whale"},
    {"question": "In what country would you find the Eiffel Tower?", "answer": "france"},
    {"question": "What is the name of the galaxy we live in?", "answer": "milky way"},
    {"question": "How many sides does a triangle have?", "answer": "three"},
    {"question": "Who is the author of the Harry Potter series?", "answer": "rowling"},
]

def get_haystack_text(url):
    """Downloads and extracts the full text content from the URL."""
    console.print(f"--- Downloading 'haystack' document from [cyan]{url}[/cyan] ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        console.print("[green]Successfully downloaded haystack content.[/green]")
        return text
    except Exception as e:
        console.print(f"[bold red]Failed to download or process haystack: {e}[/bold red]")
        return None

def main():
    """Main function to orchestrate the multi-run, randomized baseline experiment."""
    console.print(Panel("[bold cyan]Large Context Experiment: Part A - Overnight Baseline Run[/bold cyan]", border_style="green"))

    # --- Setup ---
    console.print("--- Initializing tokenizer and vLLM client ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = ChatOpenAI(
        openai_api_base=VLLM_URL,
        api_key="NOT_USED",
        model=MODEL_ID,
        temperature=0,
    )
    
    full_haystack_text = get_haystack_text(HAYSTACK_URL)
    if not full_haystack_text:
        return

    # --- NEW: Tokenize once to get the total size for logging ---
    total_tokens = tokenizer.encode(full_haystack_text)
    console.print(f"[green]Total available tokens in haystack: {len(total_tokens)}[/green]")


    all_results = []
    
    # --- NEW: Outer loop for the 5 "Grand Runs" ---
    for grand_run_num in range(1, TOTAL_GRAND_RUNS + 1):
        console.print(Panel(f"Starting Grand Run {grand_run_num} of {TOTAL_GRAND_RUNS}", border_style="magenta", padding=(1,2)))
        
        # --- NEW: Randomize the order of tests for each grand run ---
        test_configs = [("Zero Context", 0)] + [("Full Context", size) for size in CONTEXT_SIZES_TO_TEST]
        random.shuffle(test_configs)
        console.print(f"  [dim]Randomized test order for this run: {[config[1] for config in test_configs]}[/dim]")

        for test_type, context_size in test_configs:
            console.print(f"\n[bold]Testing '{test_type}' with context size: {context_size} tokens[/bold]")

            haystack = ""
            if test_type == "Full Context":
                # Use the pre-tokenized list for efficiency
                truncated_tokens = total_tokens[:context_size]
                haystack = tokenizer.decode(truncated_tokens)
                console.print(f"  [dim]Actual token count: {len(truncated_tokens)}[/dim]")

            for sub_run_num in range(1, TOTAL_SUB_RUNS + 1):
                description = f"[Grand Run {grand_run_num}, Sub-Run {sub_run_num}] {context_size}-token queries..."
                for item in track(GENERAL_KNOWLEDGE_QUESTIONS, description=description):
                    question = item["question"]
                    expected_answer = item["answer"]
                    
                    if test_type == "Zero Context":
                        prompt = ChatPromptTemplate.from_messages([("human", question)])
                    else:
                        prompt = ChatPromptTemplate.from_messages([
                            ("human", f"Based on your general knowledge, and ignoring the long text below, please answer the following question.\n\nDOCUMENT:\n{haystack}\n\nQUESTION:\n{question}")
                        ])

                    chain = prompt | llm
                    start_time = time.time()
                    response = chain.invoke({})
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    model_response_text = response.content
                    accuracy = 1 if expected_answer in model_response_text.lower() else 0
                    
                    # --- NEW: Capturing the full model response ---
                    all_results.append({
                        "grand_run": grand_run_num,
                        "sub_run": sub_run_num,
                        "test_type": test_type,
                        "context_size": context_size,
                        "question": question,
                        "response_time": response_time,
                        "accuracy": accuracy,
                        "model_response": model_response_text
                    })
        
        if grand_run_num < TOTAL_GRAND_RUNS:
            console.print(f"\n[bold blue]Grand Run {grand_run_num} complete. Pausing for 30 seconds...[/bold blue]")
            time.sleep(30)


    # --- Final Analysis and Export ---
    console.print("\n" + "="*50)
    console.print("[bold green]All experimental runs are complete. Finalizing results...[/bold green]")
    console.print("="*50 + "\n")
    results_df = pd.DataFrame(all_results)
    
    summary = results_df.groupby(['test_type', 'context_size']).agg(
        avg_response_time=('response_time', 'mean'),
        avg_accuracy=('accuracy', 'mean')
    ).reset_index().sort_values(by="context_size")

    summary_table = Table(title="Overall Baseline Performance Summary (Average of 5 Grand Runs)", show_header=True, header_style="bold magenta")
    summary_table.add_column("Test Type", style="dim")
    summary_table.add_column("Context Size (Tokens)", justify="right")
    summary_table.add_column("Avg. Response Time (s)", justify="right")
    summary_table.add_column("Avg. Accuracy", justify="right")

    for _, row in summary.iterrows():
        summary_table.add_row(
            row['test_type'],
            str(row['context_size']),
            f"{row['avg_response_time']:.4f}",
            f"{row['avg_accuracy']:.2%}"
        )
    
    console.print(summary_table)

    console.print(f"\n--- Saving all raw results to [cyan]{OUTPUT_CSV_PATH}[/cyan] ---")
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    console.print("[green]CSV file saved successfully.[/green]")
    console.print(Panel("[bold magenta]Experiment A (Overnight Run) complete![/bold magenta]"))

if __name__ == "__main__":
    main()

