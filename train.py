import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler


file_path = '/Users/tanujs/Desktop/hack/data.csv'
data = pd.read_csv(file_path)


X_columns = ['gt2', 'length', 'width', 'height', 'a_lenght', 'a_area']
y_column = 'y'


scaler = StandardScaler()
X = scaler.fit_transform(data[X_columns].to_numpy())
y = data[y_column].to_numpy()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  

num_classes = 80 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 50)  
        self.fc2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork()

try:
    model.load_state_dict(torch.load("model_weights.pth"))
    print("Loaded saved weights and biases.")
except FileNotFoundError:
    print("No saved weights found. Training will start from scratch.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  


epochs = 100000
for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y)
    
    with torch.no_grad():
        predictions = torch.argmax(output, dim=1)
        correct = (predictions == y).sum().item()
        accuracy = correct / y.size(0) * 100

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "model_weights.pth")
print("Model weights and biases saved.")
