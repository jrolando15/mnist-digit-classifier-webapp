from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image 
import numpy as np
import io 

app = Flask(__name__)



# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\Lenovo\Documents\CNN\CNN\my_cnn_model.h5')

@app.route('/')
def home():
    # Display a simple form for file upload
    return '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Upload new File</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
      <div class="container mt-5">
        <h1 class="text-center mb-4">Upload a New File</h1>
        <div class="row justify-content-center">
           <div class="col-md-6">
             <form method="post" enctype="multipart/form-data" action="/predict" class="card p-4">
               <div class="mb-3">
                 <label for="image" class="form-label">Choose an image:</label>
                 <input type="file" class="form-control" id="image" name="image" accept="image/*" required>
               </div>
               <div class="d-grid">
                 <button type="submit" class="btn btn-primary">Upload and Predict</button>
               </div>
             </form> 
           </div>
        </div>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>    
    '''

@app.route('/upload_form')
def upload_form():
    return render_template('upload_form.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return 'No image uploaded.', 400

            file = request.files['image']
            if not file:
                return 'No image uploaded.', 400

            filename = secure_filename(file.filename)
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image = image.resize((28, 28)).convert('L')  # Assuming your model expects 28x28 grayscale images
            image = np.array(image)
            image = image / 255.0  # Normalize
            image = image.reshape(1, 28, 28, 1)  # Reshape for the model

            prediction = model.predict(image)
            digit = np.argmax(prediction)

            return render_template('prediction.html', digit=digit)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # GET request: display the form again or provide information
        return render_template('upload_form.html')

if __name__ == '__main__':
    app.run(debug=True)

