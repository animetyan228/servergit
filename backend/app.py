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

CORS(app, origins=["http://localhost:3000", "http://45.12.134.146:8080"])


device = torch.device("cpu")

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем модель в CPU с нужным dtype
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)


def generate_structured_data(text):
    prompt = f"""
Ты — ИИ-секретарь. Проанализируй текст доверенности и выдели строго пять полей:

    кто_выдал — полное ФИО и организация, кто выдал доверенность. ФИО должно быть в именительном падеже, без сокращений, без лишних слов. Укажи обязательно только **одно лицо**.
    кому_выдана — полное ФИО, кому выдана доверенность. Тоже в именительном падеже, без сокращений и лишних слов.
    тема — краткое описание сути поручения (например: «получить что то», «представлять интересы»).
    дата_начала — дата начала действия доверенности в формате ДД.ММ.ГГГГ.
    дата_окончания — дата окончания действия доверенности в формате ДД.ММ.ГГГГ.

Обязательно:

   - в полях кто_выдал и кому_выдана не должно быть повторов или слияния нескольких имен.
   - каждый человек или организация должны быть представлены отдельно и корректно.
   - никаких лишних слов, без повторов, только чистые ФИО или названия.
   - ответ — только JSON-объект с указанными ключами, без комментариев и пояснений.
   - НИ В КОЕМ СЛУЧАЕ **НЕ ОТПРАВЛЯЙ КОД**, МНЕ НУЖЕН ИМЕННО **JSON-ОБЪЕКТ**
    Текст доверенности:
    
    \"\"\"{text}\"\"\"
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )

    generated_tokens = outputs[0]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(result)

    # Пытаемся извлечь JSON
    json_objects = re.findall(r'{[\s\S]*?}', result)
    for json_candidate in reversed(json_objects):
        try:
            parsed = json.loads(json_candidate)
            required_keys = {"кто_выдал", "кому_выдана", "тема", "дата_начала", "дата_окончания"}
            if required_keys.issubset(parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            continue

    return {
        "ошибка": "не удалось извлечь корректный json",
        "ответ_модели": result
    }


def ocr_pdf_with_pymupdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB)
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
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                texts = [page.extract_text() for page in pdf.pages]
            text = "\n".join(filter(None, texts))

            if len(text.strip()) < 20:
                text = ocr_pdf_with_pymupdf(pdf_bytes)
        else:
            image = Image.open(BytesIO(file.read()))
            text = pytesseract.image_to_string(image, lang='rus')

        structured_data = generate_structured_data(text)

        return jsonify({
            "распознанный_текст": text,
            "структура": structured_data
        })

    except Exception as e:
        return jsonify({"ошибка": str(e)})


@app.route("/lol", methods=["GET", "POST"])
def lol():
    return jsonify({"message": "lol"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
