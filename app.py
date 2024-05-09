from flask import Flask, request, jsonify

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np
from PIL import Image
import io

# Define the model class as in your original code snippet
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(784, 1),
        )

    def forward(self, x):
        return self.model(x)

# Create a Flask app
app = Flask(__name__)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load("model.pkl", map_location=device))
model.eval()

# Define a route for predicting cyclone intensity

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello World"
@app.route('/predict', methods=['POST'])
def predict_intensity():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert the uploaded image to a format suitable for prediction
    image_bytes = file.read() # Read the image file
    img = Image.open(io.BytesIO(image_bytes)) # Open it as a PIL image

    # Transform to tensor and resize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((250, 250)),
    ])
    img = transform(img)

    # Add batch dimension
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    # Make prediction
    with torch.no_grad():
        prediction = model(img)
        predicted_intensity = prediction.item()

    # Return the result as JSON
    return jsonify({"predicted_intensity": round(predicted_intensity, 2)})

# Print "hello world" when the app starts
print("hello world")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)