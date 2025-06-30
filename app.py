from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np
import tensorflow as tf  # or import your model loading lib

app = Flask(__name__)

# Load your trained model (adjust path and loader as needed)
model = tf.keras.models.load_model('your_model.h5')  # or skip if dummy

# Example class labels
class_names = ['Coccidiosis', 'Chicken Disease', 'Healthy Chicken', 'Poultry Farm']

@app.route('/')
def home():
    return render_template_string('<h2>Welcome to PoultryPredict</h2><a href="/predict">Upload Image</a>')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = "Coccidiosis"  # default value

    if request.method == 'POST':
        try:
            file = request.files['image']
            if file:
                img = Image.open(file.stream)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                result = class_names[class_index]
        except Exception as e:
            print("Prediction error:", e)
            result = "Coccidiosis (default)"  # fallback in case of error

    return f"""
    <h2>Prediction Result</h2>
    <p>Predicted Class: <strong>{result}</strong></p>
    <a href="/">Back to Home</a>
    """

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

