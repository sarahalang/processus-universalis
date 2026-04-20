import csv
import json
import os
import time
from openai import OpenAI

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CREDENTIAL LOADING ---
def load_env():
    # Look for .env in the same directory as this script
    script_env = os.path.join(SCRIPT_DIR, "../.env")
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

INPUT_CSV = os.path.join(SCRIPT_DIR, "unsupervised_segments_full.csv")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "extracted_opac_knowledge.json")
LIMIT_SEGMENTS = None
DELAY = 0.0 

# --- GERMAN SYSTEM PROMPT WITH FEW-SHOT EXAMPLES ---
SYSTEM_PROMPT = """
Du bist ein Experte für historische alchemistische Texte und die Extraktion von prozeduralem Wissen. 
Dein Ziel ist es, das wesentliche prozedurale Wissen aus frühneuhochdeutschen Rezeptfragmenten zu extrahieren. 

Konzentriere dich auf den **Haupt-Vorgang** (die zentrale chemische oder technische Operation) eines Segments. Vermeide es, jede einzelne Bewegung oder jeden Nebensatz als separaten Schritt zu zählen. Gruppiere stattdessen alle zugehörigen Informationen um die Kern-Handlung.

Struktur pro Schritt:
1. Haupt-Vorgang: Die zentrale Operation (z.B. "Probenahme der Erde", "Extraktion des Salzes", "Destillation").
2. Ziel-Substanz: Das primäre Material oder Produkt, um das es in diesem Vorgang geht.
3. Vollzug & Details: Eine Liste von Unter-Handlungen, Werkzeugen, Attributen, Mengen oder Spezifikationen (wie der Vorgang genau ausgeführt wird).
4. Kontext: Zeitliche, räumliche oder meteorologische Bedingungen (Wann, Wo, unter welchem Himmel).

### Beispiele für die Extraktion:

Beispiel 1:
Text: "soll mann morgensfrüh bey Sonnen-Aufgang, auf eine schöne Wiese gehen, die eine gute, fette schwartze Erden hat... darauf soll mann etliche große weit und runde Graben machen... das Gras und die Wurtzeln aber müßen vorhero alle daraus abgesondert werden"
Ausgabe:
{
  "schritte": [
    {
      "haupt_vorgang": "Probenahme / Ausgraben der Erde",
      "ziel_substanz": "fette schwartze Erde",
      "vollzug_details": [
        "große weit und runde Graben machen",
        "5 oder 4 Maßruthen breit",
        "bis an die Knie tief graben",
        "Gras und Wurzeln absondern",
        "mit dem Grabscheid ausstechen"
      ],
      "kontext": ["morgensfrüh", "Sonnen-Aufgang", "schöne Wiese", "im Monat Mai"]
    }
  ]
}

Beispiel 2 (Zwei distinkte Haupt-Vorgänge):
Text: "laßet es 24 Stunden darauf stehen, daß es das reine Saltz aus der Erden an sich ziehe, darauf laßet es unten durch das Zapfen-Loch fein sachte abtreuflen"
Ausgabe:
{
  "schritte": [
    {
      "haupt_vorgang": "Mazeration / Auslaugen",
      "ziel_substanz": "Salz aus der Erde",
      "vollzug_details": ["24 Stunden stehen lassen", "Salz an sich ziehen lassen"],
      "kontext": []
    },
    {
      "haupt_vorgang": "Abseihen / Filtrieren",
      "ziel_substanz": "Erden-Lauge",
      "vollzug_details": ["durch das Zapfen-Loch", "fein sachte abtreuflen lassen"],
      "kontext": ["in ein untergesetztes hölzernes Geschirr"]
    }
  ]
}

WICHTIG: 
- Fasse zusammenhängende Handlungen zu einem logischen "Haupt-Vorgang" zusammen.
- Behalte die historische Terminologie in den Details bei.
- Wenn ein Segment nur Theorie beschreibt, nenne den Haupt-Vorgang "Theoretische Erläuterung".
"""

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Error: API Key not set. Please update the .env file.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} nicht gefunden.")
        return
    input_path = INPUT_CSV

    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    if LIMIT_SEGMENTS:
        segments = segments[:LIMIT_SEGMENTS]
        print(f"Verarbeite die ersten {LIMIT_SEGMENTS} Segmente...")

    all_extracted_knowledge = []

    for i, seg in enumerate(segments):
        tid = seg['text_id']
        sid = seg['seg_id']
        text = seg['full_text']

        print(f"[{i+1}/{len(segments)}] Verarbeite {tid} (Seg {sid})...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extrahiere S-V-E-K Wissen aus diesem Segment:\n\n{text}"}
                ]
            )

            content = response.choices[0].message.content
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()

            raw_data = json.loads(clean_content)
            steps = raw_data.get('schritte', raw_data.get('steps', [raw_data]))
            
            all_extracted_knowledge.append({
                "text_id": tid,
                "seg_id": sid,
                "schritte": steps
            })

            # Incremental save
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_extracted_knowledge, f, indent=2, ensure_ascii=False)

            time.sleep(DELAY)

        except Exception as e:
            print(f"Fehler bei Segment {tid}-{sid}: {e}")
            continue

    print(f"\nFertig! Extrahiertes Wissen wurde in {OUTPUT_JSON} gespeichert.")

if __name__ == "__main__":
    main()
