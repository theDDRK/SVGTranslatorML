import re
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from spire.pdf.common import *
from spire.pdf import *
import fasttext
import ctypes
import torch
from datasets import Dataset

def add_id_to_text_elements(svg_contents):
    text_elements = re.findall(r"<text[^>]*>.*?</text>", svg_contents, re.DOTALL)
    print(f"Found {len(text_elements)} text elements")

    for i, text_element in enumerate(text_elements):
        id_value = f"text-{i}"
        svg_contents = svg_contents.replace(text_element, text_element.replace("<text", f"<text id=\"{id_value}\""))

    return svg_contents

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

def get_src_lang(text):
    # https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin
    model = fasttext.load_model("lid218e.bin")
    predictions = model.predict(text, k=1)
    lang_code = predictions[0][0].split("__")[-1]
    print(f"Detected source language: {lang_code} (confidence: {predictions[1][0]})")
    return lang_code

def translate_text(text_list, target_lang="en", batch_size=4):
    # Setup generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    context_prompt = (
        "You are a professional technical translator for bicycle mechanics. "
        "Translate the following sentences from English to Dutch. "
        "Keep technical terms accurate and natural. Only output the translation.\n\n"
    )

    # Create prompts
    full_prompt = context_prompt + "\n".join(f"{t}" for t in text_list)

    outputs = generator(full_prompt, max_new_tokens=512, do_sample=False)
    translation_output = outputs[0]["generated_text"]

    # Split the output into individual translations
    translated_texts = translation_output.split("\n")[len(context_prompt.split("\n")):]
    translated_texts = [text for text in translated_texts if text.strip()]

    return translated_texts

def add_translation(svg_contents, instructions):
    for instruction in instructions:
        text_id = instruction["text_id"]
        translated_content = instruction["translated_content"]
        svg_contents = re.sub(rf'(<text[^>]*id="{text_id}"[^>]*>)(.*?)</text>', lambda match: f"{match.group(1)}{translated_content}</text>", svg_contents, count=1)
    return svg_contents

def format_content(content):
    content = re.search(r'<text.*?>(.*?)</text>', content, re.DOTALL).group(1)
    content = content.replace("&#160;", " ")
    content = content.strip()
    return content

def get_instructions(svg_contents):
    text_elements = re.findall(r"<text[^>]*>.*?</text>", svg_contents, re.DOTALL)

    instructions = []

    for i in range(len(text_elements)):
        id_match = re.search(r'id="(.*?)"', text_elements[i]).group(1)
        matrix_match = re.search(r'matrix\((.*?)\)', text_elements[i])
        if matrix_match:
            x_matrix, y_matrix = matrix_match.group(1).split(" ")[-2:]
        else:
            print(f"Warning: No matrix found in text element {i}")
            x_matrix, y_matrix = "0", "0"
        content = format_content(text_elements[i])
        if len(content) > 0 and not content.isnumeric():
            instructions.append(
                {
                    "text_id": id_match,
                    "content": content,
                    "x_matrix": x_matrix,
                    "y_matrix": y_matrix,
                }
            )
    
    return [
        x for x in instructions
        # if not((float(x["x_matrix"]) >= 623 and float(x["y_matrix"]) >= 681)
        # or (float(x["x_matrix"]) < 393 and float(x["y_matrix"]) < 151))
        if (float(x["x_matrix"]) >= 25 and float(x["x_matrix"]) <= 1160)
        and (float(x["y_matrix"]) >= 30 and float(x["y_matrix"]) <= 815)
    ]

def remove_text_coordinates(svg_contents, instructions):
    for instruction in instructions:
        text_id = instruction["text_id"]
        svg_contents = re.sub(
            rf'(<text[^>]*id="{text_id}"[^>]*?)\s*x="[^"]*"([^>]*>)',
            r'\1\2',
            svg_contents
        )
        svg_contents = re.sub(
            rf'(<text[^>]*id="{text_id}"[^>]*?)\s*y="[^"]*"([^>]*>)',
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
    # Add IDs to text elements
    svg_contents = add_id_to_text_elements(svg_contents)
    # Extract instructions from the SVG contents
    instructions = get_instructions(svg_contents)
    print(f"Extracted {len(instructions)} instructions")
    print(instructions)
    # Split text elements if they contain multiple tspan elements
    # svg_contents = split_text_elements(svg_contents, instructions)
    # Extract instructions again after splitting
    # instructions = get_instructions(svg_contents)

    # Translate the text content
    translated_instructions = translate_text([x["content"] for x in instructions], target_lang)
    print(f"Translations: {translated_instructions}")
    # Replace the original text with the translated text in the SVG contents
    svg_contents = add_translation(svg_contents, [{"text_id": x["text_id"], "translated_content": translated} for x, translated in zip(instructions, translated_instructions)])
    
    # Remove x and y coordinates from instruction tspan elements
    svg_contents = remove_text_coordinates(svg_contents, instructions)
    
    return svg_contents

def convert_pdf_to_svg(pdf_file_path):
    try:
        doc = PdfDocument()
        doc.LoadFromFile(pdf_file_path)
        page_count = doc.Pages.Count
        print(f"The PDF has {page_count} pages.")
        doc.SaveToFile(pdf_file_path.replace(".pdf", ".svg"), FileFormat.SVG)
        doc.Close()
        return page_count
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while converting PDF to SVG: {e}")


if __name__ == "__main__":
    ctypes.CDLL(".venv\Lib\site-packages\spire\pdf\lib\libSkiaSharp.dll")

    file = "files/24113846_REV 00 - 24113846-2.pdf"

    if file.endswith(".pdf"):
        page_count = convert_pdf_to_svg(file)
        file = file.replace(".pdf", "")
    elif file.endswith(".svg"):
        page_count = 1
        file = file.replace(".svg", "")
    else:
        raise ValueError("File must be a PDF or SVG file.")

    target_lang = "eng_Latn"
    # model_name = "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
    # model_name = "facebook/m2m100_418M"
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    start_time = time.time()
    
    for i in range(page_count):
        if page_count != 1:
            file = f"{file}_{i + 1}"
        svg_contents = open_svg_file(file)
        svg_contents = main(svg_contents, target_lang)
        write_svg_file(file + f"_{target_lang}.svg", svg_contents)
    
    end_time = time.time()

    print("SVG file modified and saved successfully (took {:.2f} seconds)".format(end_time - start_time))