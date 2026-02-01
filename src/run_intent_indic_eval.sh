python3 -m src.run_intent_indic \
  --input_file "data/hinglish/final_jmir_data/cleaned_data_for_jmir.xlsx" \
  --output_file "output/Krutrim.json" \
  --model "krutrim-ai-labs/Krutrim-2-instruct" \
  --max_retries 3 \
  --sleep 2 \
  --max_rows 100

# Other Indic models
# --model "CohereLabs/aya-expanse-8b"
# --model "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
# --model "GenVRadmin/AryaBhatta-GemmaGenZ-Vikas-Merged"
# --model "krutrim-ai-labs/Krutrim-2-instruct"