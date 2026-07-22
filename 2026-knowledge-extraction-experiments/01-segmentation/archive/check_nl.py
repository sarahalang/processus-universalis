import xml.etree.ElementTree as ET
import os

def check_for_double_newlines(xml_path):
    if not os.path.exists(xml_path):
        xml_path = "../../" + xml_path
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    divs = root.findall('.//div')
    
    count = 0
    for div in divs:
        text = "".join(div.itertext())
        if "\n\n" in text:
            count += 1
            
    print(f"Found {count} divs with double newlines out of {len(divs)}")

if __name__ == "__main__":
    check_for_double_newlines('sammlung_aller_texte.xml')
