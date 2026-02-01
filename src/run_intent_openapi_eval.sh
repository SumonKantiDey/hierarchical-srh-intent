python3 -m src.run_intent_openapi \
  --input_file "data/hinglish/final_jmir_data/cleaned_data_for_jmir.xlsx" \
  --output_file "output/gpt-5.json" \
  --model "openai/gpt-5" \
  --max_retries 3 \
  --sleep 2 \
  --max_rows 100

# Proprietary models (via OpenRouter)
# --model "openai/gpt-4o"
# --model "openai/gpt-5"
# --model "anthropic/claude-3.5-sonnet"

# Open-weight instruction-tuned models (via OpenRouter)
# --model "meta-llama/llama-3.1-8b-instruct"
# --model "meta-llama/llama-3.3-70b-instruct"
# --model "google/gemma-2-9b-it"
# --model "google/gemma-3-27b-it"
# --model "qwen/qwen-2.5-7b-instruct"
# --model "mistralai/mixtral-8x7b-instruct"

# Indic model (via Sarvam AI)
# --model "sarvam-m"