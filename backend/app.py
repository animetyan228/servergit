from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import pytesseract
import fitz
import pdfplumber
from tasks import process_file_task

app = Flask(__name__)
CORS(app)

latest_task_id = None

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
    global latest_task_id

    if 'file' not in request.files:
        return jsonify({"ошибка": "файл не передан"})

    file = request.files['file']
    file_bytes = file.read()
    filename = file.filename.lower()

    try:
        task = process_file_task.apply_async(args=[file_bytes, filename])
        latest_task_id = task.id
        return jsonify({"Результат": "готов"})
    except Exception as e:
        return jsonify({"ошибка": str(e)})


@app.route("/result", methods=["GET"])
def get_latest_result():
    global latest_task_id

    if latest_task_id is None:
        return jsonify({"ошибка": "обработка ещё не запускалась"})

    task = process_file_task.AsyncResult(latest_task_id)
    if task.state == 'PENDING':
        return jsonify({"status": "ожидание"})
    elif task.state != 'FAILURE':
        return jsonify({"status": task.state, "результат": task.result})
    else:
        return jsonify({"status": "ошибка", "детали": str(task.info)})
