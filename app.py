import os
import uuid
from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)
INPUT_FOLDER = "input"
os.makedirs(INPUT_FOLDER, exist_ok=True)

# ---------- helper ----------
def pretty_name(raw: str) -> str:
    """Convert 'snake_case_slug' -> 'Title Case With Spaces'"""
    specials = {
        "dota": "Dota",
        "pubg": "PUBG",
        "fifa": "FIFA",
        "gta": "GTA",
        "csgo": "CS:GO",
    }
    raw_lower = raw.lower()
    if raw_lower in specials:
        return specials[raw_lower]
    return raw.replace("_", " ").title()
# ----------------------------

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(INPUT_FOLDER, filename)
            image.save(filepath)

            label, confidence = predict_image(filepath)
            os.remove(filepath)

            # use the helper here
            return render_template("display.html",
                                   label=pretty_name(label),
                                   confidence=confidence)

    # GET request: show the upload form
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)
