# Understanding User Intent in Code-Mixed Sexual and Reproductive Health Queries in Urban India: A Hierarchical Classification Approach using LLMs
This project frames Sexual and Reproductive Health (SRH) intent understanding as a two-level hierarchical classification task (topic → subtopic) and evaluates a diverse set of proprietary and open-weight LLMs under a unified experimental setup.

### 🔬 Method Overview ###
<p align="center"><img src="./figures/method.png" alt="Model Architecture" width="500"/></p>

Queries are classified using a predefined intent hierarchy covering major SRH domains.

### 🤖 Models Evaluated
All models were accessed via the OpenRouter API, with the exception of `sarvam-m`, which was accessed directly through the Sarvam AI platform.

Models were accessed via the OpenRouter API, the Sarvam AI platform (`sarvam-m`), and local GPU-based inference for selected Indic open-weight models. All models were evaluated using identical prompts and inference settings to enable fair comparison.

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
- ```CohereLabs/aya-expanse-8b```

#### Indic open-weight models
- ```ai4bharat/Airavata```
- ```Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1```
- ```GenVRadmin/AryaBhatta-GemmaGenZ-Vikas-Merged```
- ```krutrim-ai-labs/Krutrim-2-instruct``` 
- ```sarvam-m```

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
- ```run_intent_openapi.py``` – API-based models (OpenRouter and Sarvam AI)
- ```run_intent_indic.py``` – Locally executed Indic open-weight models

#### API-based inference 
```bash
python3 -m src.run_intent_openapi \
  --input_file "" \
  --output_file "" \
  --model "" \
  --max_retries 3 \
  --sleep 2 \
  --max_rows 100
```
#### Local inference
```bash
python3 -m src.run_intent_indic \
  --input_file "" \
  --output_file "" \
  --model "" \
  --max_retries 3 \
  --sleep 2 \
  --max_rows 100
```
> Note: The local GPU script requires a CUDA-enabled environment and sufficient GPU memory.

### 📊 Batch Evaluation
Batch experiments are executed using the following scripts:
```bash
chmod +x src/run_intent_openapi_eval.sh
src/run_intent_openapi_eval.sh

chmod +x src/run_intent_indic_eval.sh
src/run_intent_indic_eval.sh
```
Each script internally calls inference script (`run_intent_openapi.py` or `run_intent_indic.py`) for the configured set of models.


> ⚠️ Make sure you have appropriate API keys stored in the ```.env``` file or have a local setup for each LLM.

### 📚 Citation 
```
```