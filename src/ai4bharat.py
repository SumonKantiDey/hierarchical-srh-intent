import torch
import logging
import os
import json
import re
import time
import pandas as pd
import src.settings as settings
from .utils import hierarchy,load_existing_results,save_result_incrementally
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

logger = logging.getLogger(__name__)

MODEL = "ai4bharat/Airavata"   # Hugging Face model id
MAX_RETRIES = 2

input_file = "data/hinglish/final_jmir_data/cleaned_data_for_jmir.xlsx" 
output_json_file = "output/Airavata.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Active GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
logger.info(f"Using model: {MODEL}, with max retries: {MAX_RETRIES}, output file: {output_json_file}")

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)

def chat_wrap(user_text: str) -> str:
    return f"<|user|>\n{user_text}\n<|assistant|>\n"

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

def extract_json_obj(text: str):
     # 1) Split at the assistant marker
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>", 1)[1]

    # 2) Find the first JSON block {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in assistant output")

    candidate = text[start:end+1].strip()

    # 3) Clean up common issues
    candidate = candidate.replace("“", "\"").replace("”", "\"") \
                         .replace("’", "'").replace("‘", "'")
    
    # 4) Remove trailing commas
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # 5) Fix bad nested quotes inside Reason
    candidate = re.sub(
        r'("Reason"\s*:\s*")([^"]*?)"([^"]*?)(")',
        lambda m: f'{m.group(1)}{m.group(2)}\\"{m.group(3)}{m.group(4)}',
        candidate
    )
    #candidate = re.sub(r"\"Confidence\"\s*:\s*\"([0-9.]+)\"", r"\"Confidence\": \1", candidate)

    return json.loads(candidate)



def _slice_after_assistant(text: str) -> str:
    ASSISTANT_MARK = "<|assistant|>"
    if ASSISTANT_MARK in text:
        return text.split(ASSISTANT_MARK, 1)[1]
    return text

def _brace_block(s: str) -> str:
    """Take substring from first { to last }."""
    start = s.find("{")
    end   = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    return s[start:end+1]

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _confidence_to_number(s: str) -> str:
    return re.sub(r'"Confidence"\s*:\s*"([0-9.]+)"', r'"Confidence": \1', s)

def _sanitize_reason_by_words(candidate: str, keep_quotes: bool = False) -> str:
    """
    Finds the Reason line, strips inner quotes by tokenizing and rejoining.
    If keep_quotes=True, it escapes inner quotes instead of removing them.
    Assumes Reason is on a single line.
    """
    def repl(m):
        prefix, val = m.group(1), m.group(2).strip()

        # Preserve trailing comma if present
        trailing_comma = "," if val.endswith(",") else ""

        # Strip outer quotes if present
        if val.startswith('"'):
            val = val[1:]
        if val.endswith('",'):
            val = val[:-2]
        elif val.endswith('"'):
            val = val[:-1]

        # Tokenize, clean quotes, and rejoin
        if keep_quotes:
            # escape inner quotes instead of removing
            val = val.replace("\\", "\\\\").replace('"', '\\"')
        else:
            # remove inner quotes entirely
            tokens = val.split()
            tokens = [t.replace('"', "") for t in tokens]
            val = " ".join(tokens)
            val = val.replace("\\", "\\\\")  # keep JSON-safe backslashes

        return f'{prefix}"{val}"{trailing_comma}'

    # Match the whole Reason line (simple, one-line assumption)
    return re.sub(r'^(\s*"Reason"\s*:\s*)(.*)$', repl, candidate, flags=re.MULTILINE)

def extract_json_after_assistant_simple(text: str, keep_reason_quotes: bool = False):
    # 1) after assistant
    txt = _slice_after_assistant(text)

    # 2) JSON block
    cand = _brace_block(txt)

    # 3) clean Reason by split-and-join
    cand = _sanitize_reason_by_words(cand, keep_quotes=keep_reason_quotes)

    # 4) other minor fixes
    cand = _remove_trailing_commas(cand)
    cand = _confidence_to_number(cand)

    # 5) parse
    return json.loads(cand)

def classify_with_retry(query: str, max_new_tokens: int = 128, temperature: float = 0.0):
    prompt = create_prompt(query)
    logger.info(prompt)
    for attempt in range(MAX_RETRIES + 1):
        try:
            enc = tok(chat_wrap(prompt), return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    pad_token_id=tok.eos_token_id
                )
            full = tok.decode(out[0], skip_special_tokens=True)
            print(full)
            parsed = extract_json_after_assistant_simple(full)
            if parsed and parsed.get("Topic"):
                return {
                    "Topic": parsed.get("Topic"),
                    "Subtopic": parsed.get("Subtopic"),
                    "Confidence": parsed.get("Confidence"),
                    "Reason": parsed.get("Reason"),
                    "Raw_Output": "Not Taken"
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
    
    results = load_existing_results(output_json_file)
    processed_indexes = {r["Index"] for r in results}
    
    # Process each query
    for ind, row in df.iterrows():
        Index = row['Index']
        query = row['User Content']

        if Index in processed_indexes:
            print(f"Query already processed Index: {Index}")
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
            model=MODEL
        )
        logger.info(f"{ind=}, {Index=}, {query=}")
        time.sleep(3)  # Small delay to avoid rate limits

if __name__ == "__main__":
    main()