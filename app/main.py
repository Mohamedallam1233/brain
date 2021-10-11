from PIL import Image
import flask
import io
import numpy as np
import keras
import cv2
app = flask.Flask(__name__)
@app.route("/", methods=["POST"])
def predictt():
    data = {"success": False}
    if flask.request.method :
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            opencvImage_model_1 = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            X = cv2.resize(opencvImage_model_1, (150, 150))
            X = X.reshape((150, 150, 1))
            X = X / 255.0
            X = np.array([X])
            model = keras.models.load_model('mod2.h5')
            predictions = model.predict_classes(X)
            res = predictions[0]
            CATEGORIES = ["glioma", "meningioma", "no_tumor", "pituitary"]
            res = CATEGORIES[res]
            data["predictions"] = str(res)
            data["success"] = True
    return flask.jsonify(data)
