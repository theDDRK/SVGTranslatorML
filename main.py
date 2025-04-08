import re
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from spire.pdf.common import *
from spire.pdf import *

def split_text_elements(svg_contents, instructions):
    text_elements = re.findall(r"<text[^>]*>.*?</text>", svg_contents, re.DOTALL)
    print(f"Found {len(text_elements)} text elements")

    text_elements = [
        text_element for text_element in text_elements if any(
            instruction["tspan_id"] in text_element for instruction in instructions
        )
    ]
    print(f"Filtered {len(text_elements)} text elements with tspan id")
    
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
    with autocast('cuda'):
        translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def add_translation(svg_contents, instructions):
    for instruction in instructions:
        tspan_id = instruction["tspan_id"]
        translated_content = instruction["translated_content"]
        svg_contents = re.sub(rf'(<tspan[^>]*id="{tspan_id}"[^>]*>)(.*?)</tspan>', lambda match: f"{match.group(1)}{translated_content}</tspan>", svg_contents, count=1)
    return svg_contents

def get_instructions(svg_contents):
    tspan_matches = re.findall(
        r"<text.*?style=\"(.*?)\".*?>.*?<tspan.*?id=\"(.*?)\".*?>(.*?)</tspan>.*?</text>", 
        svg_contents, re.DOTALL
    )
    instructions = [
        {
            "tspan_id": tspan_id,
            "content": content,
            "font-size": font_size.group(1) if (font_size := re.search(r"font-size:([\d.]+)", style)) else None,
            "-inkscape-font-specification": font_spec.group(1) if (font_spec := re.search(r"-inkscape-font-specification:([^;]+)", style)) else None
        }
        for style, tspan_id, content in tspan_matches
    ]
    return [
        x for x in instructions
        if x["font-size"] and float(x["font-size"]) > 12 and "Bold" in x["-inkscape-font-specification"]
    ]

def remove_tspan_coordinates(svg_contents, instructions):
    for instruction in instructions:
        tspan_id = instruction["tspan_id"]
        svg_contents = re.sub(
            rf'(<tspan[^>]*id="{tspan_id}"[^>]*?)\s*x="[^"]*"([^>]*>)',
            r'\1\2',
            svg_contents
        )
        svg_contents = re.sub(
            rf'(<tspan[^>]*id="{tspan_id}"[^>]*?)\s*y="[^"]*"([^>]*>)',
            r'\1\2',
            svg_contents
        )
    return svg_contents

def open_svg_file(svg_file_path):
    try:
        with open(svg_file_path + ".svg", "r", encoding="utf-8") as svg_file:
            return svg_file.read()
    except FileNotFoundError:
        print(f"Error: The file '{svg_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")

def write_svg_file(svg_file_path, svg_contents):
    try:
        with open(svg_file_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_contents)
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

def main(svg_contents, target_lang):
    # Extract instructions from the SVG contents
    instructions = get_instructions(svg_contents)
    # Split text elements if they contain multiple tspan elements
    svg_contents = split_text_elements(svg_contents, instructions)
    # Extract instructions again after splitting
    instructions = get_instructions(svg_contents)

    # Translate the text content
    translated_instructions = translate_text([x["content"] for x in instructions], target_lang)
    print(f"Translations: {translated_instructions}")
    # Replace the original text with the translated text in the SVG contents
    svg_contents = add_translation(svg_contents, [{"tspan_id": x["tspan_id"], "translated_content": translated} for x, translated in zip(instructions, translated_instructions)])
    
    # Remove x and y coordinates from instruction tspan elements
    svg_contents = remove_tspan_coordinates(svg_contents, instructions)
    
    return svg_contents

def convert_pdf_to_svg(pdf_file_path):
    try:
        doc = PdfDocument()
        doc.LoadFromFile(pdf_file_path + ".pdf")
        doc.SaveToFile(pdf_file_path + ".svg", FileFormat.SVG)
        doc.Close()
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while converting PDF to SVG: {e}")


if __name__ == "__main__":
    # file = "files/test"
    # convert_pdf_to_svg(file)

    file = "10005187_REV-03-DT048336"
    target_lang = "nld"

    model_name = "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
    # model_name = "facebook/m2m100_418M"
    # model_name = "facebook/nllb-200-distilled-600M"


    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    start_time = time.time()
    
    svg_contents = open_svg_file(file)
    svg_contents = main(svg_contents, target_lang)
    write_svg_file(file + f"_{target_lang}.svg", svg_contents)
    
    end_time = time.time()

    print("SVG file modified and saved successfully (took {:.2f} seconds)".format(end_time - start_time))