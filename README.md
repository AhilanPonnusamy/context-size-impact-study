# Tools-latency-impact-on-agentic-ai
---
## This [blog](https://yyy.medium.com/@ahilanp/beyond-latency-tolerance-architecting-latency-averse-systems-for-agentic-ai-089f14a4318a) post provides a detailed explanation of this study ##

This repository provides a framework for testing the performance and accuracy of Small Language Models (SLMs) under various context lengths. The project analyzes sub-10B parameter models, including Meta-Llama-3.1-8B and Qwen2-7B, using a high-performance vLLM inference server.

Our methodology offers a holistic analysis by collecting detailed data from two distinct hardware environments: a local Apple M2 Pro chip and a cloud-based NVIDIA T4 GPU via Google Colab. The goal is to measure how an SLM's response time and retrieval accuracy are impacted as the context size increases.

While running the full test suite can take several days, this repository includes rich, pre-collected ###.csv files. This allows you to perform your own analysis immediately, exploring how these powerful SLMs behave under different conditions without the lengthy data collection process.

#### CRITICAL NOTE: vLLM is highly optimized for NVIDIA GPUs on Linux. Support for Apple Silicon (MPS) is experimental. You may encounter installation or runtime errors.

### Step 1: Create and Activate the Virtual Environment

```bash
python3.11 -m venv agentic-venv-vllm
source agentic-venv-vllm/bin/activate
```
### Step 2. Clone the Repository
Clone the project to your local machine.

```bash
git clone AhilanPonnusamy/Tools-latency-impact-on-agentic-ai
cd Tools-latency-impact-on-agentic-ai
```

### Step 3: Install PyTorch for Apple Silicon (MPS)
This is a critical first step. You must install the nightly pre-release versions of PyTorch to get the best performance and compatibility on Apple Silicon.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Step 4: Install the Remaining Dependencies
Now that PyTorch is handled, you can install the rest of the standard Python packages from our clean requirements.txt file.

```bash
pip install -r requirements.txt
```

### Step 5: Start the vLLM Server (Terminal 1)
Open a new terminal window (with the venv activated) and run the following command. This will download the Llama 3.1 model and start the vLLM server, which exposes an OpenAI-compatible API on port 8000. Leave this terminal window open for the entire experiment.

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 4096 --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser llama3_json
```

### Step 6: Run the Backend Tool Server (Terminal 2)
Open a second terminal window (with the venv activated) and start the tool server.

```bash
uvicorn mcp-tools:app --host 127.0.0.1 --port 8002
```

### Step 7: Run the Agent Logic Server (Terminal 3)
Open a third terminal window (with the venv activated) and start the agent server.

```bash
uvicorn agent-llama31:app --host 127.0.0.1 --port 8001
```

### Step 8: Run the Automated Load Test (Terminal 4)
Once all three servers are running, open a fourth terminal window (with the venv activated).

Run the following command. This single script will orchestrate the entire experiment, running for approximately 80 minutes.

```bash
python run_load_test.py
```

After the script completes, you will have two new files in your directory:

1) ```load_test_results.csv```: Contains the raw data from all 15 test runs.

2) ```latency_impact_on_throughput.png```: A line graph visualizing the final results.

