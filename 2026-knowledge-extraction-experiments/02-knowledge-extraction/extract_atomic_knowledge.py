import csv
import json
import os
import time
import re
from openai import OpenAI

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "../data/texttiling_segments.csv")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")

# --- CREDENTIAL LOADING ---
def load_env():
    script_env = os.path.join(SCRIPT_DIR, "../../.env")
    if os.path.exists(script_env):
        with open(script_env, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

load_env()

API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert in historical alchemy. Your goal is to extract knowledge units from Early Modern German recipe segments. 

CRITICAL: Return ONLY a valid JSON object. 
DO NOT include any thinking process, reasoning, or preamble. 
Start the response immediately with '{'.

### Units of Extraction:
1. **Procedural Step**: Every distinct physical laboratory action or instruction. Keep these precise and individual.
2. **Descriptive Passage**: Conceptual claims, theoretical justifications, or general commentary. Group coherent theoretical arguments into a single block.

### JSON Schema:
{
  "extracted_units": [
    {
      "unit_type": "procedural" | "descriptive",
      "normalized_intent": "Summary",
      "details": {
        "operation": "German verb (only for procedural)",
        "materials": ["substances"],
        "apparatus": ["tools"]
      },
      "context_and_theory": "Conditions or theoretical background",
      "state_dependencies": "Note on required previous states",
      "raw_source": "Verbatim German snippet"
    }
  ]
}
"""

def repair_json(raw_content):
    """Attempt to fix common JSON errors from LLMs."""
    content = raw_content.strip()
    
    # Remove potential markdown wrappers
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\n', '', content)
        content = re.sub(r'\n```$', '', content)
    
    # Try to find the JSON object if there's preamble
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    
    # Fix missing closing braces (very common on truncation)
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces > close_braces:
        content += '}' * (open_braces - close_braces)
        
    # Fix missing closing brackets
    open_brackets = content.count('[')
    close_brackets = content.count(']')
    if open_brackets > close_brackets:
        content += ']' * (open_brackets - close_brackets)
        
    return content

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Error: OpenAI API Key not found. Check your .env file.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    print(f"Loaded {len(segments)} segments for atomic extraction.")
    
    all_results = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_ids = { (r['text_id'], str(r['segment_id'])) for r in all_results }
            print(f"Found {len(processed_ids)} already processed segments.")
        except Exception as e:
            print(f"Warning: Could not load existing JSON for resume: {e}")

    remaining_segments = [s for s in segments if (s['text_id'], str(s['segment_id'])) not in processed_ids]
    print(f"Processing {len(remaining_segments)} remaining segments.")

    for i, seg in enumerate(remaining_segments):
        tid = seg['text_id']
        sid = seg['segment_id']
        text = seg['full_text']

        print(f"[{i+1}/{len(remaining_segments)}] Processing {tid} (Seg {sid})...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transform this segment into atomic units:\n\n{text}"}
                ],
                response_format={"type": "text"},
                max_tokens=8192, 
                temperature=0.1
            )

            raw_content = response.choices[0].message.content
            if raw_content is None:
                print(f"Error: Model returned None content. Finish reason: {response.choices[0].finish_reason}")
                continue
            
            # Use the repair utility
            json_str = repair_json(raw_content)
            
            try:
                result_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                # One last attempt: find the last valid '}' and cut off there
                last_brace = json_str.rfind('}')
                if last_brace != -1:
                    try:
                        result_json = json.loads(json_str[:last_brace+1])
                    except:
                        print(f"Error: Failed to parse JSON even after repair. Preview:\n{json_str[:200]}...")
                        continue
                else:
                    print(f"Error: Failed to parse JSON. Preview:\n{json_str[:200]}...")
                    continue

            steps = result_json.get('extracted_units', [])
            
            all_results.append({
                "text_id": tid,
                "segment_id": sid,
                "extracted_units": steps
            })

            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error at {tid}-{sid}: {e}")
            continue

    print(f"\nExtraction complete. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
