# SLM Context Length Performance & Accuracy Analysis
---
## This [blog](https://yyy.medium.com/@ahilanp/beyond-latency-tolerance-architecting-latency-averse-systems-for-agentic-ai-089f14a4318a) post provides a detailed explanation of this study ##

This repository provides a framework for testing the performance and accuracy of Small Language Models (SLMs) under **various context lengths**. The project analyzes sub-10B parameter models, including **Meta-Llama-3.1-8B and Qwen2-7B**, using a high-performance vLLM inference server and huggingface pipeline for comparison of vLLM prefix caching impact.

This methodology offers a holistic analysis by collecting detailed data from two distinct hardware environments: a **local Apple M2 Pro chip and a free tier cloud-based NVIDIA T4 GPU via Google Colab**. The goal is to measure how an SLM's response time and retrieval accuracy are impacted as the context size increases.

While running the full test suite can take **several days**, this repository includes rich, pre-collected data captured in (```.csv```) files. This allows you to perform your own analysis immediately, exploring how these powerful SLMs behave under different conditions without the lengthy data collection process.

#### CRITICAL NOTE: vLLM is highly optimized for NVIDIA GPUs on Linux. Support for Apple Silicon (MPS) is experimental. You may encounter installation or runtime errors.

### Step 1: Create and Activate the Virtual Environment

To begin, open your terminal. From there, create a new Python virtual environment and then activate it to ensure the project's dependencies are managed in an isolated space.

```bash
python3.11 -m venv context-venv
source context-venv/bin/activate
```
### Step 2. Clone the Repository
Clone the project to your local machine.

```bash
git clone AhilanPonnusamy/context-size-impact-study
cd context-size-impact-study
```
### Step 3: Install the  Dependencies
Install the dependent standard Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```
**login to Huggingface with a write token** using ```huggingface-cli login``` command. please ensure that you have accepted the terms and conditions for the llama3.1 model before you proceed.

### Step 4: Start the vLLM Server
With the venv activated, run the following command. This will download the Llama 3.1 model and start the vLLM server, which exposes an OpenAI-compatible API on port 8000.

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-batched-tokens 8192
```

### Step 5: Run the Automated Baseline Tests (Terminal 2)
Once all vLLM inference server is up and running, open the second terminal window (with the venv activated).

To start the experiment, run the data collection script. Before you begin, please open the file and **confirm that the correct model and output filename are set** in the configuration section. A complete test is lengthy and may take over five hours, but you can shorten this by reducing the number of test runs. Keep in mind that doing so will generate a smaller dataset for your analysis.

```bash
python baseline-test.py
```

After the script completes, you will have a new data file created in your directory.

### Step 6: Run the Automated Baseline Tests for the Second Model (Qwen) 

 Restart the vLLM server in ```Terminal 1``` with the second model (Qwen) as shown below

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-batched-tokens 8192
```
Update ```baseline-test.py``` with the Qwen model and Qwen model output filename and execute the script.

```bash
python baseline-test.py
```
After the script completes, you will have a new data file for Qwen2 model created in your directory.

### Step 7: Create a Virtual Environment for Hugging Face Pipeline Test (Terminal 3)

Open a new terminal and create a separate virtual environment for the Hugging Face pipeline test, as it requires a **different set of dependencies** than the vLLM experiments.

```bash
python3.11 -m venv hf-inference-venv
source hf-inverence-venv/bin/activate
```

### Step 8 : Install PyTorch for Apple Silicon (MPS)
This is a critical step. You must install the nightly pre-release versions of PyTorch to get the best **performance and compatibility** on Apple Silicon for this test.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Step 9 : Inatall the Dependencies and Run the Hugging Face Pipeline test 
In terminal 3 and with the newly created hugging face virtual environment activated from **```Step 7```**, install the dependent libraries

```bash
pip install -r hf-requirements.txt
```
Run the hugging face pipeline test

```bash
python hf-inference-test.py
```
After the script completes, you will have a new data file created in your directory.

### Step 10 : Run the Low Throughput vLLM test for Comparing the Performance Anomoly 
Navigate back to **Terminal 2**

With the **context-venv** activated, run the **low throughput** test

```bash
puthon lowthroughput-test.py
```

### Step 11 : Run the Comparison Data Collection scripts on Colab.
Login to **colab** and make sure you have selected T4 CPU (for free tier) or other GPU types for paid subscription. 

To run the data collection, please upload and execute the provided notebooks in sequential order using the **```File > Open Notebook > Upload```** menu. If you are using the free tier of Google Colab, it is crucial that you **download your result files immediately after a script finishes** to prevent data loss from unexpected runtime refreshes. Due to the free tier's limitations on capacity and runtime, you may need to execute the scripts on separate days. Alternatively, consider running them in **parallel using two different accounts in incognito mode** to complete the data collection more quickly.

1. Colab_Experiment_Notebook.ipynb
2. Colab_haystack_Notebook.ipynb

Download the data collection files to the project directory for analysis.

### Step 12 : Run the Graph Generation Scripts 

Run the following scripts to generate graphs for analysis
1. comparatative-analysis.py (**To accurately compare results whether between different models like Llama 3.1 and Qwen2, or between different hardware like a local Mac and a cloud GPU, it is essential that you set the correct parameters in the script's configuration section before execution.**)
2. haystack-comparative-analysis.py
3. haystack-context-length-analysis.py
