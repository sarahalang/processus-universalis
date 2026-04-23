import xml.etree.ElementTree as ET
import os

def find_indented_paragraphs(xml_path):
    if not os.path.exists(xml_path):
        xml_path = "../../" + xml_path
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    divs = root.findall('.//div')
    
    count_with_tag = 0
    count_without_tag = 0
    
    for div in divs:
        # Check start of div
        if div.text:
            lines = div.text.split('\n')
            for line in lines:
                if line.strip() and (len(line) - len(line.lstrip()) == 1):
                    count_without_tag += 1
        
        for child in div:
            if child.tail:
                lines = child.tail.split('\n')
                first_content_line = True
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        if indent == 1:
                            if first_content_line:
                                count_with_tag += 1
                            else:
                                count_without_tag += 1
                        first_content_line = False
                        
    print(f"Indented lines (Level 1) following a tag: {count_with_tag}")
    print(f"Indented lines (Level 1) NOT following a tag: {count_without_tag}")

if __name__ == "__main__":
    find_indented_paragraphs('sammlung_aller_texte.xml')
