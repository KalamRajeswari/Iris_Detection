# app.py

from flask import Flask, render_template, request, url_for
import numpy as np
from joblib import load

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load('iri.pkl')

# Dictionary to map class indices to class names and image filenames
class_dict = {
    0: ('Setosa', 'setosa.jpg'),
    1: ('Versicolor', 'versicolor.jpg'),
    2: ('Virginica', 'virginica.jpg')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare data for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(features)[0]
        class_name, image_filename = class_dict[prediction]

        image_url = url_for('static', filename=f'{image_filename}')

        return render_template('index.html', prediction=class_name, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
