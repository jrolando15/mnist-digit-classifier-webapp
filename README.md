# mnist-digit-classifier-webapp

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- Flask
- NumPy

You can install the required libraries using the following commands:
bash
pip install tensorflow
pip install flask
pip install numpy  

## Usage
1) clone the repository
git clone https://github.com/jrolando15/mnist-digit-classifier-webapp.git
cd mnist-digit-classifier-webapp

2) Run the Jupyter notebook to train the model:
   bash
   jupyter notebook
(run the mnist_digit_classifier.ipynb notebook.)

3) Start the Flask web application
   python app.py
   (Open your web browser and go to http://127.0.0.1:5000 to interact with the web app.)

## Project Structure
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

## Data Processing
The dataset is fetched from TensorFlow's Keras datasets. The images are reshaped and normalized, and the labels are one-hot encoded.

## Model Training
A Convolutional Neural Network (CNN) is defined and trained using TensorFlow. The model is trained for 9 epochs with data augmentation.

## Model Evaluation 
The model is evaluated on the test set, and the test accuracy is printed.

## Web App
The web application is built using Flask. It allows users to upload an image of a handwritten digit and get a prediction from the trained model.
   


