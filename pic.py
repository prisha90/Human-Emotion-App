# for showing only pic selection

from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("models/emotion_mobilenet.h5")

labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route("/", methods=["GET","POST"])
def index():
    emotion = ""
    confidence = ""
    user_image = ""

    if request.method == "POST":
        file = request.files["image"]

        filepath = "static/uploaded.jpg"
        file.save(filepath)
        user_image = filepath

        img = cv2.imread(filepath)
        img = cv2.resize(img,(96,96))
        img = img / 255.0
        img = img.reshape(1,96,96,3)

        pred = model.predict(img)
        idx = np.argmax(pred)

        emotion = labels[idx]
        confidence = round(float(np.max(pred)) * 100, 2)

    return render_template(
        "index1.html",
        emotion=emotion,
        confidence=confidence,
        user_image=user_image
    )

if __name__ == "__main__":
    app.run(debug=True)