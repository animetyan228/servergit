from flask import Flask, request, jsonify
from flask_cors import CORS
from tasks import process_file_task

app = Flask(__name__)
CORS(app)

latest_task_id: str | None = None

@app.route("/", methods=["POST"])
def upload():
    global latest_task_id

    if "file" not in request.files:
        return jsonify({"error": "файл не передан"})

    f      = request.files["file"]
    bytes_ = f.read()
    task   = process_file_task.apply_async(args=[bytes_, f.filename.lower()])

    latest_task_id = task.id
    return jsonify({"статус": "принят"})

@app.route("/result", methods=["GET"])
def result():
    if latest_task_id is None:
        return jsonify({"error": "обработка ещё не запускалась"})

    task = process_file_task.AsyncResult(latest_task_id)

    match task.state:
        case "PENDING":
            return jsonify({"статус": "ожидание"})
        case "STARTED":
            return jsonify({"статус": "стартует"})
        case "PROGRESS":
            return jsonify({"статус": "в работе", **task.info})
        case "FAILURE":
            return jsonify({"статус": "ошибка", "детали": str(task.info)})
        case _:
            return jsonify({"статус": "готово", "result": task.result})
