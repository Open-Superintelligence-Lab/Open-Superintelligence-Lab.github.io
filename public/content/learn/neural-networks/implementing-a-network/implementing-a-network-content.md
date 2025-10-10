---
hero:
  title: "Implementing a Network"
  subtitle: "Building Complete Neural Networks in PyTorch"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

Let's build complete, working neural networks from scratch!

## Simple Feedforward Network

```python
import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create network
model = FeedForwardNet(input_size=784, hidden_size=128, output_size=10)

# Test
x = torch.randn(32, 784)
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

## Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 2. Create model, loss, optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
def train(model, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        # Forward
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

# 4. Train
X = torch.randn(100, 10)
y = torch.randn(100, 1)
trained_model = train(model, X, y)
```

## Multi-Layer Deep Network

```python
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

model = DeepNet()
```

## Complete MNIST Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        return self.network(x)

# Create model
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total
```

## Key Takeaways

✓ **Structure:** Define model as `nn.Module`

✓ **Forward:** Implement `forward()` method

✓ **Training:** Forward → loss → backward → update

✓ **Complete pipeline:** Model + criterion + optimizer

**Quick Reference:**

```python
# Define
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(...)
    
    def forward(self, x):
        return self.layers(x)

# Train
model = MyNet()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Remember:** You can now build any neural network! 🎉
