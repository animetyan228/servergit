from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import BytesIO
from flask_cors import CORS
import pytesseract
import torch
import re
import json
import pdfplumber
import fitz

# указываем путь к tesseract
#pytesseract.pytesseract.tesseract_cmd = r'c:\program files\tesseract-ocr\tesseract.exe'

app = Flask(__name__)

CORS(app)

model_name = "thebloke/mistral-7b-instruct-v0.2-gptq" # загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


def generate_structured_data(text):
    prompt = f""" 
ты ии-секретарь. проанализируй текст доверенности и строго верни только json-объект в таком виде:

{{
  "кто_выдал": "",
  "кому_выдана": "",
  "тема": "",
  "дата_начала": "",
  "дата_окончания": ""
}}

!!! обязательно заполни поле "тема" — это то, что доверенное лицо должно сделать (например, «получить выписку», «представлять интересы» и т.п.)

вот текст доверенности:

\"\"\"{text}\"\"\"
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_tokens = outputs.sequences[0]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Ищем JSON в ответе
    json_objects = re.findall(r'{[\s\S]*?}', result)
    for json_candidate in reversed(json_objects):
        try:
            parsed = json.loads(json_candidate)
            required_keys = {"кто_выдал", "кому_выдана", "тема", "дата_начала", "дата_окончания"}
            if required_keys.issubset(set(parsed.keys())):
                return parsed
        except json.JSONDecodeError:
            continue

    return {
        "ошибка": "не удалось извлечь корректный json",
        "ответ_модели": result
    }


# достаем текст из pdf с помощью pymupdf и ocr
def ocr_pdf_with_pymupdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB)  # явный RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang='rus')
        full_text.append(text)
    return "\n".join(full_text)

@app.route("/", methods=["POST"])
def extract_and_structure():
    if 'file' not in request.files:
        return jsonify({"ошибка": "файл не передан"})

    file = request.files['file']
    filename = file.filename.lower()

    try:
        if filename.endswith(".pdf"):
            pdf_bytes = file.read()
            # попытка достать текст из pdf напрямую если с текстом
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                texts = [page.extract_text() for page in pdf.pages]
            text = "\n".join(filter(None, texts))

            # если текста нет(<20), делаем через ocr
            if len(text.strip()) < 20:
                text = ocr_pdf_with_pymupdf(pdf_bytes)
        else:
            # если не pdf, тогда картинка
            image = Image.open(BytesIO(file.read()))
            text = pytesseract.image_to_string(image, lang='rus')

        structured_data = generate_structured_data(text)

        # конечный ответ
        return jsonify({
            "распознанный_текст": text,
            "структура": structured_data
        })

    # вывод ошибки если ты лох и не запустилось
    except Exception as e:
        return jsonify({"ошибка": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)