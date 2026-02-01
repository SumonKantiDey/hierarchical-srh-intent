# Understanding User Intent in Code-Mixed Sexual and Reproductive Health Queries in Urban India: A Hierarchical Classification Approach using LLMs
This project frames Sexual and Reproductive Health (SRH) intent understanding as a two-level hierarchical classification task (topic → subtopic) and evaluates a diverse set of proprietary and open-weight LLMs under a unified experimental setup.

### 🔬 Method Overview ###
<p align="center"><img src="./figures/method.png" alt="Model Architecture" width="500"/></p>
Queries are classified using a predefined intent hierarchy covering major SRH domains.

### 🤖 Models Evaluated
All models were accessed via the OpenRouter API, with the exception of `sarvamai/sarvam-m`, which was accessed directly through the Sarvam AI platform.


####  Proprietary models
- ```openai/gpt-4o```
- ```openai/gpt-5```
- ```anthropic/claude-3.5-sonnet```

#### Open-weight instruction-tuned models
- ```meta-llama/llama-3.1-8b-instruct```
- ```meta-llama/llama-3.3-70b-instruct```
- ```google/gemma-2-9b-it```
- ```google/gemma-3-27b-it```
- ```qwen/qwen-2.5-7b-instruct```
- ```mistralai/mixtral-8x7b-instruct```
- ```sarvamai/sarvam-m```

All models are evaluated using identical prompts and inference settings to enable fair comparison.

### 📁 Repository Structure ###
- `./confidence_intervals` - Scripts and notebooks for confidence interval estimation.
- `./error_analysis` - Aggregated error analysis across evaluated models.
- `./figures` - Figures used in the paper and reports.
- `./src` - Source code and runnable scripts, including scripts for computing the code-mixing index (CMI).
- `./results` - Aggregated model predictions and evaluation outputs.

### ⚙️ Setup ### 
First, clone this repo and move to the directory. Then, install the necessary libraries. Also, the following commands can be used:
```bash
$ git clone https://github.com/SumonKantiDey/hierarchical-srh-intent.git
$ cd hierarchical-srh-intent/ 
$ pip install -r requirements.txt
```
### Copy the example environment file ###
```
$ cp .env.example .env
```


### ▶️ Running Model Inference 
The run_intent.py script performs hierarchical intent classification for a single model specified via command-line arguments.

```bash
python3 -m src.run_intent \
  --input_file "" \
  --output_file "" \
  --model "" \
  --max_retries 3 \
  --sleep 2 \
  --max_rows 100
```

### 📊 Batch Evaluation

To run experiments across multiple models and random seeds, use the provided shell script:
```bash
chmod +x src/eval.sh
src/eval.sh
```
This script internally calls ```run_intent.py``` for each configured model.

> ⚠️ Make sure you have appropriate API keys stored in the ```.env``` file or have a local setup for each LLM.

### 📚 Citation 
```
```