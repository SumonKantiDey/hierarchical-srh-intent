import torch
import logging
import os
import json
import re
import time
import pandas as pd
import src.settings as settings
from .utils import hierarchy,load_existing_results,save_result_incrementally, parse_args
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

args = parse_args()
logger = logging.getLogger(__name__)

input_file = args.input_file
output_json_file = args.output_file

MAX_RETRIES = args.max_retries

model_id = args.model

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Active GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
logger.info(f"Using model: {model_id}, with max retries: {MAX_RETRIES}, output file: {output_json_file}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    logger.info(f"{model_id} Model loaded successfully.")
except Exception as e:
    logger.info(f"Error loading model: {str(e)}")
    exit()

def create_prompt(query: str) -> str:
    return f"""
Classify the following Hinglish (Romanized Hindi + English) query into exactly ONE topic and ONE subtopic from the intent hierarchy below.

### Intent Hierarchy:
{json.dumps(hierarchy, indent=4, ensure_ascii=False)}

**Query (Hinglish):** "{query}"

**Output Format:**
Return your answer inside a JSON code block like this:

```json
{{
  "Topic": "<selected_topic>",
  "Subtopic": "<selected_subtopic_from_that_topic>",
  "Confidence": <number between 0.0 and 1.0>,
  "Reason": "<short reason>"
}}
```
# Rules:
1. Select ONLY ONE topic and ONE subtopic.
2. The subtopic MUST belong to the selected topic.
3. Confidence MUST be a decimal number between 0.0 and 1.0.
4. Reason MUST be a short sentence (max 20 words).
5. Output MUST be valid JSON inside a JSON code block.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def build_messages(query):
	return [
        {"role": "system", "content": "You are an SRH intent classification system. Follow instructions exactly."},
        {"role": "user", "content": create_prompt(query)}
    ]

def extract_json_from_markdown(output: str) -> dict:
    """
    Extract JSON from a markdown-formatted string.
    Works for both ```json ... ``` blocks and plain JSON text.
    """
    match = re.search(r"json\s*(\{.*?\})\s*", output, re.DOTALL)
    # match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    else:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {}

def classify_with_retry(query):
    messages = build_messages(query)
    # print(messages)
    for attempt in range(MAX_RETRIES + 1):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            if model_id == "krutrim-ai-labs/Krutrim-2-instruct":
                inputs.pop("token_type_ids", None) 

            # print(" ====== ",inputs["input_ids"].shape)
            output = model.generate(
                **inputs,
                max_new_tokens=200, 
                temperature=0.1,
                do_sample=False
            )
            # response = tokenizer.decode(output[0], skip_special_tokens=True)
            # decode only the generated continuation
            prompt_len = inputs["input_ids"].shape[-1]
            raw_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True) # outputs[0] is basically [prompt + answer]
            parsed = extract_json_from_markdown(raw_output)
            # logger.info(raw_output)
            if parsed and parsed.get("Topic"):
                return {
                    "Topic": parsed.get("Topic"),
                    "Subtopic": parsed.get("Subtopic"),
                    "Confidence": parsed.get("Confidence"),
                    "Reason": parsed.get("Reason"),
                    "Raw_Output": raw_output
                }
            else:
                raise ValueError("Parsed output is empty or missing Topic/Subtopic.") # will retry the same query 
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait_time = 1.5 * (attempt + 1)
                logger.info(f"[Retry {attempt+1}] Error: {e} — retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {
                "Topic": None,
                "Subtopic": None,
                "Confidence": None,
                "Reason": None,
                "Raw_Output": f"[ERROR] {e}"
            }
# --------------------
# PROCESS CSV
# --------------------
def main():
    # query = "Emergency garbhnirodhak dawaai ke alawa aur kya options hain?"
    # classification = classify_with_retry(query)
    # print(classification)

    # Load dataset
    df = pd.read_excel(input_file)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    
    results = load_existing_results(output_json_file)
    processed_indexes = {r["Index"] for r in results}
    
    # Process each query
    for ind, row in df.iterrows():
        Index = row['Index']
        query = row['User Content']

        if Index in processed_indexes:
            logger.info(f"Query already processed Index: {Index}")
            continue
    
        result = classify_with_retry(query)

        topic = result["Topic"]
        subtopic = result["Subtopic"]
        confidence = result["Confidence"]
        reason = result["Reason"]
        raw_output = result['Raw_Output']

        save_result_incrementally(
            existing_results=results,
            output_file=output_json_file, 
            Index=Index,
            query=query,
            topic=topic,
            subtopic=subtopic,
            confidence=confidence,
            reason=reason,
            raw_output=raw_output,
            model=model_id
        )
        logger.info(f"{ind=}, {Index=}, {query=}")
        time.sleep(args.sleep)  # Small delay to avoid rate limits

if __name__ == "__main__":
    main()