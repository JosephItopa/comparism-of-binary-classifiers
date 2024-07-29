# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

# Convert data to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define single-layer perceptron model
class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize model, loss function and optimizer
model = SingleLayerPerceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predictions and loss
with torch.no_grad():
    y_pred_perceptron = model(X_tensor).numpy().flatten()
loss_perceptron = log_loss(y, y_pred_perceptron)

print("Single-Layer Perceptron Loss:", loss_perceptron)