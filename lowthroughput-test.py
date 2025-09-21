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
MODEL_ID = "Qwen/Qwen2-7B-Instruct" 
HAYSTACK_URL = "http://www.paulgraham.com/cities.html"
CONTEXT_SIZES_TO_TEST = [500, 1000] # Focusing on the key comparison
SLEEP_BETWEEN_QUERIES_SECONDS = 20 # The crucial variable for this test
TOTAL_GRAND_RUNS = 3
OUTPUT_CSV_PATH = "experiment_D_low_throughput_results.csv"

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

def get_haystack_text(url, tokenizer):
    """Downloads and extracts the full text content from the URL."""
    console.print(f"--- Downloading 'haystack' document from [cyan]{url}[/cyan] ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        tokens = tokenizer.encode(text)
        console.print(f"[green]Successfully prepared haystack with {len(tokens)} tokens.[/green]")
        return tokens
    except Exception as e:
        console.print(f"[bold red]Failed to prepare haystack: {e}[/bold red]")
        return None

def main():
    """Main function to orchestrate the low-throughput test."""
    console.print(Panel("[bold cyan]Large Context Experiment: Part D - Low-Throughput 'Hack' Test[/bold cyan]", border_style="green"))

    # --- Setup ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = ChatOpenAI(openai_api_base=VLLM_URL, api_key="NOT_USED", model=MODEL_ID, temperature=0)
    
    full_haystack_tokens = get_haystack_text(HAYSTACK_URL, tokenizer)
    if not full_haystack_tokens:
        return

    all_results = []
    
    for grand_run_num in range(1, TOTAL_GRAND_RUNS + 1):
        console.print(Panel(f"Starting Grand Run {grand_run_num} of {TOTAL_GRAND_RUNS}", border_style="magenta"))
        
        test_configs = [("Zero Context", 0)] + [("Full Context", size) for size in CONTEXT_SIZES_TO_TEST]
        random.shuffle(test_configs)

        for test_type, context_size in test_configs:
            console.print(f"\n[bold]Testing '{test_type}' with context size: {context_size} tokens[/bold]")
            
            haystack = ""
            if test_type == "Full Context":
                haystack = tokenizer.decode(full_haystack_tokens[:context_size])

            for item in track(GENERAL_KNOWLEDGE_QUESTIONS, description=f"[Run {grand_run_num}] {context_size}-token queries..."):
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
                
                model_response_text = response.content
                accuracy = 1 if expected_answer in model_response_text.lower() else 0

                # --- KEY CHANGE: Add a sleep AFTER the query for context-heavy trials ---
                if test_type == "Full Context":
                    time.sleep(SLEEP_BETWEEN_QUERIES_SECONDS)

                # --- UPDATED: Capturing detailed information for analysis ---
                all_results.append({
                    "run": grand_run_num,
                    "test_type": test_type,
                    "context_size": context_size,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_response": model_response_text,
                    "response_time": end_time - start_time,
                    "accuracy": accuracy
                })

    # --- Final Analysis ---
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(['test_type', 'context_size']).agg(
        avg_response_time=('response_time', 'mean'),
        avg_accuracy=('accuracy', 'mean')
    ).reset_index()
    
    summary_table = Table(title="Low-Throughput Performance & Accuracy Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Test Type", style="dim")
    summary_table.add_column("Context Size (Tokens)")
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
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    console.print(f"[green]Full results saved to [cyan]{OUTPUT_CSV_PATH}[/cyan][/green]")
    console.print(Panel("[bold magenta]Experiment D complete![/bold magenta]"))

if __name__ == "__main__":
    main()

