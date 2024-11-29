import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = 'data1.csv'

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {file_path}. Please collect data first using the GUI.")

# Prepare the test data
X_columns = ['weight', 'hull_length', 'hull_width', 'hull_height', 'hole_length', 'hole_area']
scaler = StandardScaler()
X = scaler.fit_transform(data[X_columns].to_numpy())

X = torch.tensor(X, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 50)  
        self.fc2 = nn.Linear(50, 80)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork()

model.load_state_dict(torch.load("model_weights.pth"))
model.eval() 

with torch.no_grad():
    output = model(X)  
    predictions = torch.argmax(output, dim=1)  

print(f"Time(in min) in which ship shinks: {predictions[-1].tolist()} min")
