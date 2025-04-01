import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

svg_file_path = "10005187_REV-03-DT048336.svg"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-bible-big-mul-mul")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-bible-big-mul-mul")

def split_text_elements(svg_contents):
    text_elements = re.findall(r"<text[^>]*>.*?</text>", svg_contents, re.DOTALL)
    print(f"Found {len(text_elements)} text elements")
    
    for i in range(len(text_elements)):
        original_text_element = text_elements[i]
        tspan_elements = re.findall(r"<tspan.*?>.*?</tspan>", text_elements[i], re.DOTALL)
        
        if len(tspan_elements) > 1:
            id_match = re.search(r'id="(.*?)"', text_elements[i]).group(1)
            matrix_match = re.search(r'matrix\((.*?)\)', text_elements[i]).group(1)
            x_matrix, y_matrix = matrix_match.split(",")[-2:]
            
            for j in range(1, len(tspan_elements)):
                x_tspan = re.search(r'x="([^"]+)"', tspan_elements[j]).group(1)
                y_tspan = re.search(r'y="([^"]+)"', tspan_elements[j]).group(1)
                new_x_matrix = float(x_matrix) + float(x_tspan.split(" ")[0]) if x_tspan else 0
                new_y_matrix = float(y_matrix) - float(y_tspan.split(" ")[0]) if y_tspan else 0
                new_matrix = f"matrix({matrix_match.split(',')[0]},{matrix_match.split(',')[1]},{matrix_match.split(',')[2]},{matrix_match.split(',')[3]},{new_x_matrix},{new_y_matrix})"
                
                text_elements[i] = text_elements[i].replace(tspan_elements[j], "")
                style_attribute = re.search(r'style="(.*?)"', text_elements[i]).group(1)
                text_elements[i] += f"\n<text id=\"{id_match}-{j}\" style=\"{style_attribute}\" transform=\"{new_matrix}\">{tspan_elements[j]}</text>"
            
            svg_contents = svg_contents.replace(original_text_element, text_elements[i])
    
    return svg_contents

def translate_text(text_list, target_lang="en"):
    text_list = [">>" + target_lang + "<< " + text for text in text_list]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def add_translation(svg_contents, instructions):
    for instruction in instructions:
        tspan_id = instruction["tspan_id"]
        translated_content = instruction["translated_content"]
        svg_contents = re.sub(rf'(<tspan[^>]*id="{tspan_id}"[^>]*>)(.*?)</tspan>', lambda match: f"{match.group(1)}{translated_content}</tspan>", svg_contents, count=1)
    return svg_contents


def extract_tspan_contents(svg_file_path, target_lang):
    try:
        with open(svg_file_path, "r", encoding="utf-8") as svg_file:
            svg_contents = svg_file.read()
            svg_contents = split_text_elements(svg_contents)
            
            tspan_matches = re.findall(
                r"<text.*?style=\"(.*?)\".*?>.*?<tspan.*?id=\"(.*?)\".*?>(.*?)</tspan>.*?</text>", 
                svg_contents, re.DOTALL
            )
            
            extracted_data = []
            for style, tspan_id, content in tspan_matches:
                font_size = re.search(r"font-size:([\d.]+)", style)
                font_spec = re.search(r"-inkscape-font-specification:([^;]+)", style)
                extracted_data.append({
                    "tspan_id": tspan_id,
                    "content": content,
                    "font-size": font_size.group(1) if font_size else None,
                    "-inkscape-font-specification": font_spec.group(1) if font_spec else None
                })
            
            extracted_data.sort(key=lambda x: (x["-inkscape-font-specification"], -float(x["font-size"]) if x["font-size"] else 0))
            instructions = [x for x in extracted_data if x["font-size"] and float(x["font-size"]) > 12 and "Bold" in x["-inkscape-font-specification"]]

            translated_instructions = translate_text([x["content"] for x in instructions], target_lang)
            print(f"Translations: {translated_instructions}")
            svg_contents = add_translation(svg_contents, [{"tspan_id": x["tspan_id"], "translated_content": translated} for x, translated in zip(instructions, translated_instructions)])
            
            with open("extracted_data.json", "w", encoding="utf-8") as json_file:
                json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)
            with open("instructions.json", "w", encoding="utf-8") as json_file:
                json.dump(instructions, json_file, ensure_ascii=False, indent=4)
            
            modified_svg_contents = re.sub(r'(<tspan[^>]*?)\s*x="[^"]*"([^>]*>)', r'\1\2', svg_contents)
            modified_svg_contents = re.sub(r'(<tspan[^>]*?)\s*y="[^"]*"([^>]*>)', r'\1\2', modified_svg_contents)
            
            modified_svg_file_path = svg_file_path.replace(".svg", "_modified.svg")
            with open(modified_svg_file_path, "w", encoding="utf-8") as modified_svg_file:
                modified_svg_file.write(modified_svg_contents)
    
    except FileNotFoundError:
        print(f"Error: The file '{svg_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

extract_tspan_contents(svg_file_path, target_lang="nld")