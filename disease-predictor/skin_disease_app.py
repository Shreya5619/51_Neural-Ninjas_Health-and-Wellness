from flask import Flask, request, render_template, jsonify
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import cv2 as cv
import numpy as np
from torchvision.datasets import ImageFolder
from werkzeug.utils import secure_filename
from sd_recognizer import CNN_Model, CustomDataset

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the transform for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=r'Datasets/skin disease recognizer/train', transform=transform)
class_names = dataset.classes
dataset= CustomDataset(dataset=dataset)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load your CNN model for skin disease prediction
model_0 = CNN_Model(input_shape=3, hidden_units=60, output_shape=len(class_names)).to(device)
model_0.load_state_dict(torch.load("Skin Disease Recognizer.pth"))
model_0.eval()

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the HTML form (with custom name)
@app.route('/')
def index():
    return render_template('skin_disease_form.html')  # Serve the renamed HTML page

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file from the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)  # Save in the 'uploads' directory
            file.save(file_path)

            # Load and preprocess the image
            img = Image.open(file_path)
            img = transform(img).unsqueeze(0)  # Add batch dimension

            # Make prediction using the CNN model
            with torch.no_grad():
                outputs = model_0(img)
                _, predicted_idx = torch.max(outputs, 1)

            predicted_disease = class_names[predicted_idx.item()]

            return jsonify({"predicted_disease": predicted_disease})

        else:
            return jsonify({"error": "Invalid file format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run the Flask application
    app.run(debug=True)
