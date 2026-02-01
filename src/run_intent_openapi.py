import os
import logging
import json
import re
import time
import pandas as pd
from openai import OpenAI
import src.settings as settings
from .utils import hierarchy,load_existing_results,save_result_incrementally, parse_args
from dotenv import load_dotenv

load_dotenv()

args = parse_args() 
logger = logging.getLogger(__name__)

input_file = args.input_file
output_json_file = args.output_file 

MODEL = args.model 
MAX_RETRIES = args.max_retries 

if MODEL == "sarvam-m":
    base_url = os.getenv("sarvam_base_url")
    api_key = os.getenv("sarvam_api_key")
else:
    base_url = os.getenv("openrouter_base_url")
    api_key = os.getenv("openrouter_api_key")   

logger.info(f"Using model: {MODEL}, with max retries: {MAX_RETRIES}, output file: {output_json_file}") 
# --------------------
# OPENROUTER CLIENT
# --------------------
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

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

def extract_json_from_markdown(output: str) -> dict:
    """
    Extract JSON from a markdown-formatted string.
    Works for both ```json ... ``` blocks and plain JSON text.
    """
    match = re.search(r"json\s*(\{.*?\})\s*", output, re.DOTALL)
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

# --------------------
# CLASSIFICATION WITH RETRIES
# --------------------
def classify_with_retry(question: str):
    prompt = create_prompt(question)
    # print("prompt = ",prompt)
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an SRH intent classification system. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            raw_output = completion.choices[0].message.content.strip()
            parsed = extract_json_from_markdown(raw_output)
            if parsed and parsed.get("Topic"):
                return {
                    "Topic": parsed.get("Topic"),
                    "Subtopic": parsed.get("Subtopic"),
                    "Confidence": parsed.get("Confidence"),
                    "Reason": parsed.get("Reason"),
                    "Raw_Output": raw_output
                }
            else:
                logger.error("Parsed output is empty or missing Topic/Subtopic.")
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
            model=MODEL
        )
        logger.info(f"{ind=}, {Index=}, {query=}")
        time.sleep(args.sleep)  # Small delay to avoid rate limits

if __name__ == "__main__":
    main()
