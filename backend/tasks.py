import re
import json
import torch
from PIL import Image
from io import BytesIO
import pytesseract
import fitz
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM
from celery import Celery

celery_app = Celery("tasks", broker="redis://redis:6379/0", backend="redis://redis:6379/0")

device = torch.device("cpu")

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)


def ocr_pdf(file_bytes, filename):
    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            texts = [page.extract_text() for page in pdf.pages]
        text = "\n".join(filter(None, texts))

        if len(text.strip()) < 20:
            # Fallback на MuPDF OCR
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            full_text = []
            for page in doc:
                pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                full_text.append(pytesseract.image_to_string(img, lang="rus"))
            text = "\n".join(full_text)
    else:
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang="rus")

    return text


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
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    json_objects = re.findall(r'{[\s\S]*?}', result)
    for json_candidate in reversed(json_objects):
        try:
            parsed = json.loads(json_candidate)
            required_keys = {"кто_выдал", "кому_выдана", "тема", "дата_начала", "дата_окончания"}
            if required_keys.issubset(parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"ошибка": "не удалось извлечь корректный json", "ответ_модели": result}


@celery_app.task(name="process_file_task")
def process_file_task(file_bytes, filename):
    print("OCR")
    text = ocr_pdf(file_bytes, filename)
    print("OCR end")
    print("JSON")
    result = generate_structured_data(text)
    print("JSON end")
    return {
        "распознанный_текст": text,
        "структура": result
    }
