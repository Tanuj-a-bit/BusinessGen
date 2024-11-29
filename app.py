
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler

# Flask app setup
app = Flask(__name__)

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 50)  # 6 inputs (features)
        self.fc2 = nn.Linear(50, 80)  # 80 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = NeuralNetwork()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# StandardScaler for input normalization
scaler = StandardScaler()
scaler.mean_ = np.array([50, 200, 40, 20, 30, 100])  # Example means (adjust as per training data)
scaler.scale_ = np.array([10, 50, 10, 5, 10, 50])  # Example scales (adjust as per training data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        weight = float(request.form['weight'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        position = float(request.form['position'])
        hole_length = float(request.form['hole_length'])
        hole_area = float(request.form['hole_area'])

        # Prepare data
        features = np.array([[weight, length, width, position, hole_length, hole_area]])
        features = scaler.transform(features)  # Normalize features
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Prediction
        with torch.no_grad():
            output = model(features_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Return prediction
        return render_template('result.html', prediction=predicted_class)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)