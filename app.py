from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import time

app = Flask(__name__)

model = tf.keras.models.load_model("malaria_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    start = time.time()

    img = load_img(filepath, target_size=(128,128))
    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    confidence = round(float(prediction)*100,2)

    if prediction > 0.5:
        result = "UNINFECTED"
    else:
        result = "PARASITIZED"
        confidence = 100-confidence

    processing_time = round(time.time() - start,2)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        img_path=filepath,
        time=processing_time
    )


if __name__ == "__main__":
    app.run(debug=True)  