import json
import os
import csv
import collections
import re

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_JSON = os.path.join(SCRIPT_DIR, "../data/labeled_segments.json")
SEGMENTS_CSV = os.path.join(SCRIPT_DIR, "../data/texttiling_segments.csv")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "../data/document_trajectories.html")

def highlight_passages(full_text, concepts, label_to_color):
    sorted_units = sorted(concepts, key=lambda x: len(x['raw']), reverse=True)
    processed_text = full_text
    replacements = {}
    for i, unit in enumerate(sorted_units):
        raw_snippet = unit['raw'].strip()
        if not raw_snippet or raw_snippet not in processed_text: continue
        base_color = label_to_color[unit['cluster_label']]
        highlight_color = base_color.replace("hsl", "hsla").replace(")", ", 0.15)")
        placeholder = f"[[HIGHLIGHT_{i}]]"
        tag_open = f'<span class="text-highlight" title="Intent: {unit["intent"]}" style="background: {highlight_color}; border-bottom: 2px solid {base_color}; cursor:pointer;" onclick="event.stopPropagation(); inspectConcept(\'{unit["cluster_label"]}\')">'
        replacements[placeholder] = f"{tag_open}{raw_snippet}</span>"
        processed_text = processed_text.replace(raw_snippet, placeholder, 1)
    for placeholder, html in replacements.items():
        processed_text = processed_text.replace(placeholder, html)
    return processed_text

def generate_navigable_report():
    if not os.path.exists(LABELED_JSON) or not os.path.exists(SEGMENTS_CSV):
        print(f"Error: Required files not found.")
        return

    with open(LABELED_JSON, 'r', encoding='utf-8') as f:
        atomic_data = json.load(f)
    
    atomic_map = collections.defaultdict(list)
    unique_labels = set()
    for unit in atomic_data:
        atomic_map[(unit['text_id'], str(unit['segment_id']))].append(unit)
        unique_labels.add(unit['cluster_label'])

    label_to_color = {}
    for i, label in enumerate(sorted(list(unique_labels))):
        hue = int(360 * i / len(unique_labels))
        label_to_color[label] = f"hsl({hue}, 70%, 45%)"

    docs = collections.defaultdict(list)
    with open(SEGMENTS_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row['text_id']
            sid = row['segment_id']
            units = atomic_map.get((tid, sid), [])
            highlighted_html = highlight_passages(row['full_text'], units, label_to_color)
            unique_segment_labels = sorted(list(set(u['cluster_label'] for u in units)))
            docs[tid].append({
                'sid': sid,
                'html_text': highlighted_html,
                'unique_labels': unique_segment_labels,
                'concepts': units # Keep for interpretation list
            })

    json_payload = json.dumps(atomic_data)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Alchemical Reader: Navigable View</title>
        <style>
            :root {{ --bg: #fdfdfb; --accent: #8b4513; --border: #e0e0e0; }}
            body {{ font-family: 'Georgia', serif; margin: 0; display: flex; height: 100vh; overflow: hidden; background: var(--bg); }}
            
            #sidebar {{ width: 80px; background: #2c3e50; display: flex; flex-direction: column; align-items: center; padding: 20px 0; overflow-y: auto; }}
            .nav-link {{ color: #bdc3c7; font-size: 11px; margin-bottom: 15px; cursor: pointer; text-decoration: none; font-family: sans-serif; }}
            .nav-link:hover {{ color: white; }}

            #doc-explorer {{ width: 50%; overflow-y: auto; padding: 0 40px 40px 40px; border-right: 1px solid var(--border); line-height: 1.8; scroll-behavior: smooth; }}
            #concept-inspector {{ width: 42%; overflow-y: auto; padding: 30px; background: white; box-shadow: -5px 0 15px rgba(0,0,0,0.05); }}
            
            .doc-section {{ position: relative; }}
            .doc-header {{ 
                position: sticky; top: 0; background: var(--bg); padding: 20px 0; z-index: 100;
                font-family: sans-serif; font-size: 22px; font-weight: bold; color: var(--accent);
                border-bottom: 2px solid var(--accent); margin-bottom: 30px;
            }}
            
            .segment-box {{ 
                background: white; border: 1px solid #f0f0f0; border-radius: 8px; padding: 25px; margin-bottom: 40px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.03); transition: outline 0.3s;
            }}
            
            .text-content {{ font-size: 17px; color: #222; margin-bottom: 20px; }}
            
            .concept-gallery {{ border-top: 1px solid #f5f5f5; padding-top: 15px; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }}
            .concept-chip {{ 
                padding: 4px 12px; border-radius: 20px; color: white; font-family: sans-serif; font-size: 11px; 
                font-weight: bold; cursor: pointer; text-transform: uppercase;
            }}
            
            .interpretation-item {{ font-family: sans-serif; border-left: 3px solid #eee; padding-left: 12px; margin-bottom: 12px; }}
            .interp-label {{ font-size: 10px; font-weight: bold; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }}
            .interp-intent {{ font-size: 13px; color: #444; display: block; }}
            .interp-context {{ font-size: 11px; color: #888; font-style: italic; }}

            .text-highlight {{ transition: filter 0.2s; padding: 1px 0; }}
            .text-highlight:hover {{ filter: brightness(0.9); }}

            .occurrence-card {{ background: #f9f9f9; padding: 15px; margin-bottom: 15px; border-radius: 6px; border: 1px solid #eee; }}
            .jump-btn {{ 
                display: inline-block; margin-top: 10px; font-family: sans-serif; font-size: 11px; 
                color: #3498db; text-decoration: underline; cursor: pointer; 
            }}
            
            ::-webkit-scrollbar {{ width: 6px; }}
            ::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 10px; }}
        </style>
    </head>
    <body>

    <div id="sidebar">
        <div style="color:white; font-weight:bold; margin-bottom:30px; font-size:10px;">DOCS</div>
        {"".join([f'<a class="nav-link" href="#doc-{tid}">{tid}</a>' for tid in sorted(docs.keys())])}
    </div>

    <div id="doc-explorer">
        <div id="docs-container"></div>
    </div>

    <div id="concept-inspector">
        <div id="inspector-content">
            <div class="inspector-header">
                <h2>Konzept-Inspektor</h2>
                <p>Wählen Sie eine Stelle links aus.</p>
            </div>
        </div>
    </div>

    <script>
        const atomicData = {json_payload};
        const labelColors = {json.dumps(label_to_color)};
        
        function renderExplorer() {{
            const container = document.getElementById('docs-container');
            const docs = {json.dumps(docs)};
            
            Object.keys(docs).sort().forEach(tid => {{
                const section = document.createElement('div');
                section.className = 'doc-section';
                section.id = `doc-${{tid}}`;
                
                const header = document.createElement('div');
                header.className = 'doc-header';
                header.innerText = `Dokument: ${{tid}}`;
                section.appendChild(header);
                
                docs[tid].forEach(seg => {{
                    const box = document.createElement('div');
                    box.className = 'segment-box';
                    box.id = `seg-${{tid}}-${{seg.sid}}`;
                    
                    let chips = '';
                    seg.unique_labels.forEach(lbl => {{
                        chips += `<div class="concept-chip" style="background: ${{labelColors[lbl]}}" 
                                       onclick="inspectConcept('${{lbl}}')">${{lbl}}</div>`;
                    }});
                    
                    box.innerHTML = `
                        <div class="text-content">${{seg.html_text}}</div>
                        <div class="concept-gallery">
                            ${{chips}}
                        </div>
                    `;
                    section.appendChild(box);
                }});
                container.appendChild(section);
            }});
        }}
        
        function inspectConcept(label) {{
            const inspector = document.getElementById('inspector-content');
            const matches = atomicData.filter(d => d.cluster_label === label);
            
            let html = `
                <div class="inspector-header">
                    <span class="concept-chip" style="background: ${{labelColors[label]}}; font-size:14px; padding:5px 15px;">
                        ${{label}}
                    </span>
                    <h3>Belege im Korpus: ${{matches.length}}</h3>
                </div>
            `;
            
            matches.forEach(m => {{
                html += `
                    <div class="occurrence-card">
                        <strong>${{m.text_id}}</strong> | Segment ${{m.segment_id}}<br>
                        <div style="font-style:italic; border-left:3px solid #ddd; padding-left:10px; margin:10px 0; color:#444; font-size:14px;">"${{m.raw}}"</div>
                        <div style="font-size:13px; color:#222; margin-top:5px;"><b>Intent:</b> ${{m.intent}}</div>
                        <div style="font-size:11px; color:#666; margin-top:2px;"><b>Context:</b> ${{m.context}}</div>
                        <div class="jump-btn" onclick="jumpTo('${{m.text_id}}', '${{m.segment_id}}')">→ Zum Text springen</div>
                    </div>
                `;
            }});
            inspector.innerHTML = html;
            inspector.parentElement.scrollTop = 0;
        }}
        
        function jumpTo(tid, sid) {{
            const target = document.getElementById(`seg-${{tid}}-${{sid}}`);
            if (target) {{
                target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                target.style.outline = '3px solid #3498db';
                setTimeout(() => target.style.outline = 'none', 3000);
            }}
        }}
        
        renderExplorer();
    </script>
    </body>
    </html>
    """
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Report updated with interpretations saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_navigable_report()
