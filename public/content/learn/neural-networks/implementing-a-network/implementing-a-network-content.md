---
hero:
  title: "Implementing a Network"
  subtitle: "Building Complete Neural Networks in PyTorch"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 10 min read"
---

Let's build complete, working neural networks from scratch! By the end of this lesson, you'll understand every line of code needed to create and train a neural network.

## Understanding the Architecture First

Before diving into code, let's understand what we're building. A **feedforward neural network** is the simplest type of network where information flows in one direction: input → hidden layers → output. Think of it like a assembly line where each station (layer) processes the data before passing it to the next.

### The Three Essential Components

Every neural network has three key parts:

1. **Input layer**: Receives raw data
2. **Hidden layer(s)**: Extracts patterns and features
3. **Output layer**: Makes predictions

Let's see how this translates to code.

## Building Your First Network

We'll start simple - a network with just one hidden layer. Here's the structure:

```
Input (784 features) → Hidden (128 neurons) → Output (10 classes)
```

This is perfect for classifying images (like MNIST digits: 28×28 = 784 pixels → 10 digit classes).

### Step 1: Import Libraries

```python
import torch
import torch.nn as nn
```

Simple! PyTorch gives us everything we need.

### Step 2: Define the Network Class

```python
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
```

**What's happening here:**
- `nn.Module`: Base class for all neural networks in PyTorch
- `__init__`: Initialize layers (like setting up the assembly line)
- `fc1`: First "fully connected" layer (784 → 128)
- `fc2`: Second layer (128 → 10)

### Step 3: Define Forward Pass

```python
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply first layer + activation
        x = self.fc2(x)               # Apply second layer
        return x
```

**The data flow:**
1. Input goes through `fc1`: 784 → 128 dimensions
2. ReLU activation adds non-linearity
3. Goes through `fc2`: 128 → 10 dimensions
4. Returns raw scores (logits) for each class

### Step 4: Create and Test

Now let's bring it to life:

```python
# Create network
model = FeedForwardNet(input_size=784, hidden_size=128, output_size=10)

# Test with dummy data
x = torch.randn(32, 784)  # 32 images, 784 pixels each
output = model(x)

print(output.shape)  # torch.Size([32, 10])
```

**What we're seeing:**
- Input: 32 samples with 784 features each
- Output: 32 samples with 10 class scores each
- The network processes all 32 images in parallel!

## The Complete Training Pipeline

Now that we have a network, how do we train it? Training requires four components working together like a feedback loop. Let's build them step by step.

### Component 1: The Model

First, let's define a simple regression network. Notice how we use `nn.Sequential` here - it's a cleaner way to stack layers:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),  # Input: 10 features → 20 hidden
            nn.ReLU(),           # Non-linearity
            nn.Linear(20, 1)     # Hidden: 20 → 1 output
        )
    
    def forward(self, x):
        return self.layers(x)  # Sequential applies layers in order
```

### Component 2: Loss Function and Optimizer

Now we need to define how to measure error and how to improve:

```python
model = Net()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
```

**What each does:**
- **Loss function**: Measures how wrong predictions are
- **Optimizer**: Updates weights to reduce loss
- **Learning rate (0.001)**: How big each update step is

### Component 3: The Training Loop

This is where the magic happens! Let's break down each step:

```python
def train(model, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        # STEP 1: Forward pass - make predictions
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
```

**Forward pass:** Run data through the network to get predictions, then calculate how wrong they are.

```python
        # STEP 2: Backward pass - calculate gradients
        optimizer.zero_grad()  # Clear old gradients (important!)
        loss.backward()         # Compute gradients via backpropagation
```

**Backward pass:** PyTorch automatically calculates how to adjust each weight.

```python
        # STEP 3: Update weights
        optimizer.step()  # Apply the calculated gradients
        
        # Print progress
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model
```

**Weight update:** The optimizer adjusts weights in the direction that reduces loss.

### Component 4: Run Training

Finally, let's put it all together:

```python
# Create some random data
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # 100 target values

# Train the model
trained_model = train(model, X, y, epochs=100)
```

**The training cycle:**
```
Epoch 0: Loss = 1.2345
Epoch 20: Loss = 0.8234
Epoch 40: Loss = 0.5123
Epoch 60: Loss = 0.3456
Epoch 80: Loss = 0.2345
```

See how the loss decreases? That's learning in action!

## Building Deeper Networks

One hidden layer is good, but multiple hidden layers can learn more complex patterns. Let's build a deep network:

### The Architecture

Let's create a 4-layer network that progressively reduces dimensions:

```
784 → 512 → 256 → 128 → 10
```

Each layer extracts higher-level features!

```python
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define all layers
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 10)
```

### Adding Dropout for Regularization

**Dropout** randomly turns off neurons during training to prevent overfitting:

```python
        self.dropout = nn.Dropout(0.2)  # Drop 20% of neurons
```

### The Forward Pass

Now let's see the data flow through all layers:

```python
    def forward(self, x):
        # Layer 1: 784 → 512
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        
        # Layer 2: 512 → 256
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        
        # Layer 3: 256 → 128
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        
        # Layer 4: 128 → 10 (no activation/dropout on output)
        x = self.layer4(x)
        return x

# Create the model
model = DeepNet()
```

**Why no activation on the final layer?** 
- For classification, we'll use CrossEntropyLoss which includes softmax
- For regression, we want raw values

## Real-World Example: MNIST Digit Classification

Let's build a complete, production-ready network for classifying handwritten digits. This example includes everything you'd use in a real project:

### Step 1: Define the Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),    # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),     # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)       # Output layer (10 digits)
        )
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten 28x28 images to 784 vector
        return self.network(x)
```

**Architecture choices:**
- 128 → 64 neurons: Gradually compress information
- Dropout: Prevents overfitting (memorizing training data)
- 10 outputs: One score for each digit (0-9)

### Step 2: Setup Training Components

```python
model = MNISTNet()
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Step 3: Training Function

Let's create a function that trains for one epoch (one pass through all data):

```python
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()  # Set to training mode (enables dropout)
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
```

**What's happening:**
- `model.train()`: Tells the model we're training (important for dropout)
- Loop through batches of data
- Calculate loss for each batch

```python
        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()         # Calculate gradients
        optimizer.step()        # Update weights
        
        total_loss += loss.item()
```

**The three magic lines:** zero → backward → step. This is the core of training!

```python
    return total_loss / len(dataloader)  # Average loss
```

### Step 4: Evaluation Function

We need to check how well our model performs on test data:

```python
def evaluate(model, dataloader):
    model.eval()  # Set to evaluation mode (disables dropout)
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't calculate gradients (saves memory)
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)  # Get class with highest score
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total  # Accuracy
```

**Key differences from training:**
- `model.eval()`: Disables dropout
- `torch.no_grad()`: Saves memory by not tracking gradients
- We care about accuracy, not loss

### Putting It All Together

```python
# Example usage
# Assume train_loader and test_loader are already created

for epoch in range(10):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    
    # Evaluate
    test_accuracy = evaluate(model, test_loader)
    
    print(f"Epoch {epoch+1}/10")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2%}")
```

**Expected output:**
```
Epoch 1/10
  Train Loss: 0.5234
  Test Accuracy: 85.23%
Epoch 2/10
  Train Loss: 0.2156
  Test Accuracy: 92.45%
...
Epoch 10/10
  Train Loss: 0.0523
  Test Accuracy: 97.84%
```

See the improvement over time? That's your network learning!

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
