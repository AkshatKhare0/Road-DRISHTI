from flask import Flask, render_template, request

import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", error="Please choose a file first.")

    files = {
        "file": (uploaded_file.filename, uploaded_file.stream, uploaded_file.mimetype)
    }

    response = requests.post(f"{FASTAPI_URL}/detect/image", files=files)

    if response.status_code != 200:
        return render_template(
            "index.html",
            error=f"Inference service error: {response.status_code}"
        )

    result = response.json()

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)