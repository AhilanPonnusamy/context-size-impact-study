import os
import time
import pandas as pd
import requests
import random
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
# --- NEW: Import libraries for local HF pipeline ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup

# --- Configuration ---
# --- UPDATED: We no longer need the VLLM URL ---
MODEL_ID = "Qwen/Qwen2-7B-Instruct" 
HAYSTACK_URL = "http://www.paulgraham.com/cities.html"
CONTEXT_SIZES_TO_TEST = [0, 1000] # Focusing on the core comparison
TOTAL_RUNS = 3 # A shorter run to quickly validate the hypothesis
OUTPUT_CSV_PATH = "experiment_E_hf_pipeline_results.csv"

console = Console()

# --- The 25 General Knowledge Questions ---
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

def get_haystack_text(url, tokenizer, max_tokens):
    """Downloads, extracts, and truncates text content."""
    console.print(f"--- Downloading 'haystack' document from [cyan]{url}[/cyan] ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        tokens = tokenizer.encode(text)
        truncated_tokens = tokens[:max_tokens]
        haystack = tokenizer.decode(truncated_tokens)
        console.print(f"[green]Successfully prepared {len(truncated_tokens)}-token haystack.[/green]")
        return haystack
    except Exception as e:
        console.print(f"[bold red]Failed to prepare haystack: {e}[/bold red]")
        return None

def main():
    """Main function to run the Hugging Face Pipeline comparison."""
    console.print(Panel("[bold cyan]Large Context Experiment: Part E - Hugging Face Pipeline Test[/bold cyan]", border_style="green"))

    # --- NEW: Setup for local Hugging Face Pipeline ---
    console.print("--- Initializing local model and tokenizer (this will take a while)... ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
    llm = HuggingFacePipeline(pipeline=pipe)
    console.print("[green]Local Hugging Face pipeline initialized successfully.[/green]")

    
    haystack = get_haystack_text(HAYSTACK_URL, tokenizer, 1000)
    if not haystack:
        return

    all_results = []
    
    # --- Simplified Test Loop ---
    for run_num in range(1, TOTAL_RUNS + 1):
        console.print(Panel(f"Starting Run {run_num} of {TOTAL_RUNS}", border_style="yellow"))
        
        # --- Trial 1: Zero Context ---
        for item in track(GENERAL_KNOWLEDGE_QUESTIONS, description=f"[Run {run_num}] Zero Context Queries..."):
            prompt = ChatPromptTemplate.from_messages([("human", item["question"])])
            question = item["question"]
            expected_answer = item["answer"]
            chain = prompt | llm
            start_time = time.time()
            response = chain.invoke({})
            model_response_text = response
            accuracy = 1 if expected_answer in model_response_text.lower() else 0           
            end_time = time.time()
            response_time = end_time - start_time
            #all_results.append({"run": run_num, "test_type": "Zero Context", "response_time": end_time - start_time})
            # --- NEW: Capturing the full model response ---
            all_results.append({
                "grand_run": 1,
                "sub_run": run_num,
                "test_type": "Zero Context",
                "context_size": 0,
                "question": question,
                "response_time": response_time,
                "accuracy": accuracy,
                "model_response": model_response_text
            })
            
        # --- Trial 2: 1000-Token Context ---
        for item in track(GENERAL_KNOWLEDGE_QUESTIONS, description=f"[Run {run_num}] 1000-Token Context Queries..."):
            prompt = ChatPromptTemplate.from_messages([
                ("human", f"DOCUMENT:\n{haystack}\n\nQUESTION:\n{item['question']}")
            ])
            question = item["question"]
            expected_answer = item["answer"]
            chain = prompt | llm
            start_time = time.time()
            response = chain.invoke({})
            end_time = time.time()
            response_time = end_time - start_time
            model_response_text = response

            accuracy = 1 if expected_answer in model_response_text.lower() else 0           
            
            #all_results.append({"run": run_num, "test_type": "500-Token Context", "response_time": end_time - start_time})
            all_results.append({
                "grand_run": 1,
                "sub_run": run_num,
                "test_type": "1000-Token Context",
                "context_size": 1000,
                "question": question,
                "response_time": response_time,
                "accuracy": accuracy,
                "model_response": model_response_text
            })
    # --- Final Analysis ---
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('test_type')['response_time'].mean().reset_index()
    
    summary_table = Table(title="Hugging Face Pipeline Performance Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Test Type", style="dim")
    summary_table.add_column("Avg. Response Time per Query (s)", justify="right")
    
    for _, row in summary.iterrows():
        summary_table.add_row(row['test_type'], f"{row['response_time']:.4f}")
    
    console.print(summary_table)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    console.print(f"[green]Full results saved to [cyan]{OUTPUT_CSV_PATH}[/cyan][/green]")
    console.print(Panel("[bold magenta]Experiment E complete![/bold magenta]"))

if __name__ == "__main__":
    main()
