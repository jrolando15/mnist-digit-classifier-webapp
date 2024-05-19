# mnist-digit-classifier-webapp

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- Flask
- NumPy

You can install the required libraries using the following commands:
```bash 
pip install tensorflow
pip install flask
pip install numpy  
```
## Usage
1) clone the repository
```bash
git clone https://github.com/jrolando15/mnist-digit-classifier-webapp.git
cd mnist-digit-classifier-webapp
```

3) Run the Jupyter notebook to train the model:
```bash
jupyter notebook
```
(run the mnist_digit_classifier.ipynb notebook.)

4) Start the Flask web application
```bash
    python app.py
```
   (Open your web browser and go to http://127.0.0.1:5000 to interact with the web app.)

## Project Structure
```bash
mnist-digit-classifier-webapp/
│
├── mnist_digit_classifier.ipynb  # Jupyter notebook with the code
├── app.py                        # Flask web application
├── templates/
│   ├── index.html                # HTML template for the welcome page
│   ├── prediction.html           # HTML template for displaying predictions
│   └── upload_form.html          # HTML template for the image upload form
├── static/
│   └── style.css                 # CSS for the web app
├── README.md                     # Project README file
└── requirements.txt              # List of dependencies
```

## Data Processing
The dataset is fetched from TensorFlow's Keras datasets. The images are reshaped and normalized, and the labels are one-hot encoded.

```bash 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

## Model Training
A Convolutional Neural Network (CNN) is defined and trained using TensorFlow. The model is trained for 9 epochs with data augmentation.

```bash 
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
     rotation_range=10,
     width_shift_range=0.1,
     height_shift_range=0.1,
     zoom_range=0.1,
     horizontal_flip=True
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = datagen.flow(train_images, train_labels, batch_size=64)
model.fit(train_generator, epochs=9, validation_data=(test_images, test_labels))
```
## Model Evaluation 
The model is evaluated on the test set, and the test accuracy is printed.
```bash
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test accuracy: {test_acc}')
```

## Web Application
The web application is built using Flask. It allows users to upload an image of a handwritten digit and get a prediction from the trained model.
   
app.py
```bash
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('my_cnn_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload_form', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        img = request.files['image']
        img_path = 'static/' + img.filename
        img.save(img_path)
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        return render_template('prediction.html', digit=digit, img_path=img_path)
    return render_template('upload_form.html')

if __name__ == '__main__':
    app.run(debug=True)
```

index.html
```bash 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Welcome</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-3">Welcome to the Digit Prediction App</h1>
        <a href="/upload_form" class="btn btn-primary">Upload Image</a>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

prediction.html
```bash 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Predicted Digit: {{ digit }}</h1>
        <a href="/upload_form" class="btn btn-info">Try another image</a>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

upload_form.html
```bash 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Image</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-3">Upload Image for Prediction</h1>
        <form action="/upload_form" method="post" enctype="multipart/form-data" class="mb-3">
            <div class="mb-3">
                <label for="imageUpload" class="form-label">Choose an image:</label>
                <input type="file" class="form-control" id="imageUpload" name="image" accept="image/*" required>
            </div>
            <button type="submit" class="btn btn-primary">Upload and Predict</button>
        </form>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```


## License
