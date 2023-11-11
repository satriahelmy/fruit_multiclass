from flask import Flask
import os

import numpy as np
from flask import Blueprint, render_template, request
from keras.models import load_model
from keras.preprocessing import image

from module import allowed_file

app = Flask(__name__)
# Load the pre-trained model
model_path = "models/fruit_multiclass.h5"
model = load_model(model_path)

class_names = ['freshapples', 'freshbanana', 'freshcucumber',
               'freshguava', 'freshlime', 'freshokra', 'freshorange',
               'freshpomegranate', 'freshpotato', 'freshtomato',
               'rottenapples', 'rottenbanana', 'rottencucumber',
               'rottenguava', 'rottenlime', 'rottenokra', 'rottenorange',
               'rottenpomegranate', 'rottenpotato', 'rottentomato']

@app.route("/", methods=["GET", "POST"])
def multiclass_classification():
    result = None  # Initialize result as None

    if request.method == "POST":
        # Handle the image upload and prediction here
        image_data = request.files['image']
        if image_data and allowed_file(image_data.filename):
            # Save the uploaded image to the "static/uploads/" directory
            upload_folder = "static/uploads/"
            os.makedirs(upload_folder, exist_ok=True)
            image_path = os.path.join(upload_folder, image_data.filename)
            image_data.save(image_path)

            # Load and preprocess the uploaded image for prediction
            img = image.load_img(image_path, target_size=(150, 150))
            img = image.img_to_array(img)

            # Make a prediction
            result = make_prediction(img)

        return render_template("pages/multiclass_classification.html",
                               result=result,
                               uploaded_image_path=image_path)
    else:
        return render_template("pages/multiclass_classification.html")


def make_prediction(image):
    # Ensure the image has the correct shape and type for the model
    image = np.expand_dims(image, axis=0)

    image = np.vstack([image])

    # Make the prediction
    prediction = model.predict(image)

    print(prediction)
    predicted_class = class_names[np.argmax(prediction[0])]
    return predicted_class
    

if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))