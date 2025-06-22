import re
import json
import torch
from io import BytesIO
from PIL import Image
import pytesseract
import fitz
import pdfplumber
from celery import Celery
from transformers import AutoTokenizer, AutoModelForCausalLM

celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)
celery_app.conf.update(
    task_track_started=True,
    worker_concurrency=1,
    task_time_limit=900,
)

_MODEL_NAME = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
_tokenizer = None
_model     = None
_DEVICE    = torch.device("cpu")


def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            use_fast=False,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": _DEVICE},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    return _tokenizer, _model


def ocr_pdf(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        text = "\n".join(filter(None, pages)).strip()
        if len(text) < 20:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            ocr_pages = []
            for p in doc:
                pix = p.get_pixmap(dpi=300, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ocr_pages.append(pytesseract.image_to_string(img, lang="rus"))
            text = "\n".join(ocr_pages)
    else:
        img = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(img, lang="rus")
    return text

_KEYS = {"кто_выдал", "кому_выдана", "тема", "дата_начала", "дата_окончания"}

_SYSTEM_MSG = (
    "Ты — ИИ‑секретарь. Твоя задача: из входного текста доверенности «как есть» "
    "выделить ровно пять полей и вернуть только JSON без лишних слов.\n\n"
    "Поля:\n"
    "кто_выдал — одно ФИО или официальное название организации (именительный).\n"
    "кому_выдана — одно ФИО (именительный).\n"
    "тема — коротко, 3–7 слов, о сути поручения.\n"
    "дата_начала — ДД.ММ.ГГГГ, пусто если нет даты.\n"
    "дата_окончания — ДД.ММ.ГГГГ или пусто.\n\n"
    "Строгое требование: верни только JSON‑объект с указанными ключами, "
    "без Markdown, кода и пояснений."
    "Строго используй **точно** эти ключи:\n"
    "кто_выдал, кому_выдана, тема, дата_начала, дата_окончания."
)


def make_json(text: str) -> dict:
    tokenizer, model = get_model()

    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {
            "role": "user",
            "content": f"Текст доверенности:\n\n\"\"\"\n{text}\n\"\"\"",
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

    candidates = re.findall(r"\{[\s\S]*?\}", raw)
    if candidates:
        best = max(candidates, key=len)
        try:
            parsed = json.loads(best)
            if _KEYS.issubset(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"ошибка": "не удалось распарсить JSON", "сырые_данные": raw}

@celery_app.task(bind=True, name="process_file_task")
def process_file_task(self, file_bytes: bytes, filename: str) -> dict:
    """OCR → LLM → структурированный JSON"""
    self.update_state(state="PROGRESS", meta={"step": "OCR"})
    text = ocr_pdf(file_bytes, filename)

    self.update_state(state="PROGRESS", meta={"step": "LLM"})
    structure = make_json(text)

    return {
        "распознанный_текст": text,
        "структура": structure,
    }
