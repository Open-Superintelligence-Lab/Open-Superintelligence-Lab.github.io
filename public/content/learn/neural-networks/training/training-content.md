---
hero:
  title: "Training & Optimization"
  subtitle: "Making Neural Networks Learn Effectively"
  tags:
    - "🧠 Neural Networks"
    - "⏱️ 16 min read"
---

# Training & Optimization

## The Training Process

Training a neural network is an iterative process of adjusting weights to minimize the loss function. The goal is to find the optimal set of parameters that make accurate predictions on both **training and unseen data**.

![Training Loop](training-loop.png)

## Gradient Descent: The Foundation

Gradient descent is the fundamental optimization algorithm:

```
1. Start with random weights
2. Calculate loss on data
3. Compute gradients (how to adjust weights)
4. Update weights in opposite direction of gradient
5. Repeat until convergence
```

### Mathematical Formula

```
θ_new = θ_old - α · ∇L(θ)
```

Where:
- `θ` = parameters (weights and biases)
- `α` = learning rate
- `∇L` = gradient of loss

Think of it as **rolling down a hill** to find the lowest point (minimum loss)!

![Gradient Descent](gradient-descent.png)

## Variants of Gradient Descent

### 1. Batch Gradient Descent

Uses **entire dataset** for each update:

```python
for epoch in range(num_epochs):
    # Compute loss on ALL data
    predictions = forward_pass(X_train, weights)
    loss = compute_loss(predictions, y_train)
    
    # Compute gradients using all data
    gradients = backward_pass(X_train, y_train, weights)
    
    # Single update per epoch
    weights -= learning_rate * gradients
```

**Pros:**
- ✅ Stable updates
- ✅ Guaranteed convergence (for convex problems)

**Cons:**
- ❌ Very slow for large datasets
- ❌ Requires entire dataset in memory
- ❌ Can get stuck in local minima

### 2. Stochastic Gradient Descent (SGD)

Updates weights after **each training example**:

```python
for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    
    for i in indices:
        # Use single example
        x, y = X_train[i], y_train[i]
        
        prediction = forward_pass(x, weights)
        loss = compute_loss(prediction, y)
        gradients = backward_pass(x, y, weights)
        
        # Update after each example
        weights -= learning_rate * gradients
```

**Pros:**
- ✅ Much faster iterations
- ✅ Can escape local minima (noise helps!)
- ✅ Works with large datasets

**Cons:**
- ❌ Noisy updates
- ❌ Can oscillate around minimum
- ❌ Harder to parallelize

### 3. Mini-Batch Gradient Descent ⭐ (Most Popular)

Best of both worlds! Uses **small batches** (32, 64, 128, 256):

```python
batch_size = 64

for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    
    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        # Forward pass on batch
        predictions = forward_pass(X_batch, weights)
        loss = compute_loss(predictions, y_batch)
        
        # Backward pass on batch
        gradients = backward_pass(X_batch, y_batch, weights)
        
        # Update weights
        weights -= learning_rate * gradients
```

**Pros:**
- ✅ Good balance between speed and stability
- ✅ Efficient GPU utilization
- ✅ More stable than SGD
- ✅ Faster than batch GD

**Cons:**
- ❌ One more hyperparameter (batch size)

![GD Variants Comparison](gd-variants.png)

## Advanced Optimizers

### 1. Momentum 🏃

Accumulates a **velocity** term to accelerate in consistent directions:

```python
velocity = 0
beta = 0.9  # momentum coefficient

for epoch in range(num_epochs):
    gradients = compute_gradients()
    
    # Update velocity
    velocity = beta * velocity + (1 - beta) * gradients
    
    # Update weights using velocity
    weights -= learning_rate * velocity
```

**Why it works:**
- Accelerates in valleys
- Dampens oscillations
- Helps escape plateaus

**Analogy:** A ball rolling down a hill gains momentum!

### 2. RMSprop

Adapts learning rate **per parameter** based on recent gradients:

```python
cache = 0
beta = 0.9

for epoch in range(num_epochs):
    gradients = compute_gradients()
    
    # Update cache (exponential moving average of squared gradients)
    cache = beta * cache + (1 - beta) * gradients**2
    
    # Update weights with adaptive learning rate
    weights -= learning_rate * gradients / (np.sqrt(cache) + 1e-8)
```

**Why it works:**
- Different learning rates for each parameter
- Larger steps for parameters with small gradients
- Smaller steps for parameters with large gradients

**Great for:** Recurrent neural networks

### 3. Adam (Adaptive Moment Estimation) ⭐ (Most Popular)

Combines **momentum** and **RMSprop**:

```python
m = 0  # First moment (mean)
v = 0  # Second moment (variance)
beta1 = 0.9
beta2 = 0.999

for epoch in range(num_epochs):
    gradients = compute_gradients()
    
    # Update moments
    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * gradients**2
    
    # Bias correction
    m_hat = m / (1 - beta1**epoch)
    v_hat = v / (1 - beta2**epoch)
    
    # Update weights
    weights -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
```

**Why it works:**
- Combines best of momentum and RMSprop
- Adaptive learning rates
- Bias correction for early iterations
- Works well in practice

**Default choice** for most deep learning tasks!

![Optimizers Comparison](optimizers-comparison.png)

## Learning Rate Strategies

### 1. Fixed Learning Rate
```python
learning_rate = 0.001  # Constant throughout training
```

Simple but often suboptimal.

### 2. Learning Rate Decay

Gradually reduce learning rate:

```python
# Step decay
initial_lr = 0.01
drop_rate = 0.5
epochs_drop = 10

lr = initial_lr * (drop_rate ** (epoch // epochs_drop))

# Exponential decay
lr = initial_lr * np.exp(-decay_rate * epoch)

# 1/t decay
lr = initial_lr / (1 + decay_rate * epoch)
```

### 3. Learning Rate Scheduling

```python
# Cosine annealing
import math

def cosine_schedule(epoch, total_epochs, lr_max, lr_min=0):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * epoch / total_epochs)
    )
```

### 4. Warm-up + Decay

```python
def lr_schedule(epoch, warmup_epochs=5, initial_lr=0.001):
    if epoch < warmup_epochs:
        # Linear warm-up
        return initial_lr * (epoch / warmup_epochs)
    else:
        # Cosine decay
        return cosine_schedule(
            epoch - warmup_epochs,
            total_epochs - warmup_epochs,
            initial_lr
        )
```

![Learning Rate Schedules](lr-schedules.png)

## Key Hyperparameters

### 1. Learning Rate (α)

**Most important hyperparameter!**

```python
# Too high: divergence
lr = 1.0  # Loss explodes ❌

# Too low: very slow training
lr = 0.00001  # Takes forever ❌

# Just right: fast and stable
lr = 0.001  # Good starting point ✅
```

**Finding the right learning rate:**
- Start with 0.001 or 0.0001
- Use learning rate finder
- Monitor training loss

### 2. Batch Size

```python
# Small batches (8-32)
# + More noise → can escape local minima
# - Slower, less stable

# Medium batches (64-128) ⭐
# + Good balance
# + Efficient GPU usage

# Large batches (256-1024)
# + Faster training (fewer updates)
# + More stable
# - Can lead to poor generalization
# - Requires more memory
```

**Rule of thumb:** Start with 32 or 64

### 3. Number of Epochs

```python
# Too few epochs
epochs = 5  # Underfitting ❌

# Too many epochs
epochs = 1000  # Overfitting ❌

# Use early stopping ✅
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(max_epochs):
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        save_model()
    else:
        counter += 1
        
    if counter >= patience:
        print("Early stopping!")
        break
```

### 4. Optimizer Parameters

```python
# Adam parameters
optimizer = Adam(
    learning_rate=0.001,  # Step size
    beta1=0.9,            # Momentum decay (usually 0.9)
    beta2=0.999,          # RMSprop decay (usually 0.999)
    epsilon=1e-8          # Numerical stability
)

# SGD with momentum
optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9          # Usually 0.9 or 0.95
)
```

## Training Best Practices

### 1. Data Preparation
```python
# Normalize inputs
X = (X - X.mean()) / X.std()

# Or use min-max scaling
X = (X - X.min()) / (X.max() - X.min())
```

### 2. Weight Initialization
```python
# Xavier/Glorot initialization (for sigmoid/tanh)
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

# He initialization (for ReLU)
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

### 3. Regularization
```python
# L2 regularization (weight decay)
loss = mse_loss + lambda_reg * np.sum(weights**2)

# Dropout (randomly zero out neurons)
if training:
    mask = (np.random.rand(*activations.shape) > dropout_rate)
    activations = activations * mask / (1 - dropout_rate)
```

### 4. Batch Normalization
```python
# Normalize activations in each layer
z_norm = (z - z.mean()) / np.sqrt(z.var() + epsilon)
z_scaled = gamma * z_norm + beta  # Learnable parameters
```

### 5. Monitoring Training

```python
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch in range(num_epochs):
    # Training
    train_loss, train_acc = train_epoch()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    # Validation
    val_loss, val_acc = validate()
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Check for overfitting
    if val_loss > train_loss * 1.2:
        print("Warning: Possible overfitting!")
```

![Training Curves](training-curves.png)

## Common Issues and Solutions

### 1. Loss Not Decreasing
**Problem:** Loss stays constant or increases

**Solutions:**
- ✅ Check learning rate (try 0.001, 0.0001)
- ✅ Verify data preprocessing
- ✅ Check for bugs in forward/backward pass
- ✅ Try different weight initialization

### 2. Training Loss Decreases, Validation Loss Increases
**Problem:** Overfitting

**Solutions:**
- ✅ Add regularization (L2, dropout)
- ✅ Reduce model complexity
- ✅ Get more training data
- ✅ Use data augmentation
- ✅ Early stopping

### 3. Loss Explodes (NaN)
**Problem:** Numerical instability

**Solutions:**
- ✅ Lower learning rate
- ✅ Use gradient clipping
- ✅ Check for division by zero
- ✅ Use batch normalization

### 4. Training Too Slow
**Problem:** Takes forever to converge

**Solutions:**
- ✅ Increase learning rate
- ✅ Use Adam instead of SGD
- ✅ Increase batch size
- ✅ Use GPU/TPU acceleration

## Complete Training Example

```python
import numpy as np

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 100
patience = 10

# Initialize optimizer
m = v = 0
beta1, beta2 = 0.9, 0.999

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    
    epoch_loss = 0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        # Forward pass
        y_pred = forward(X_batch, weights)
        loss = compute_loss(y_pred, y_batch)
        
        # Backward pass
        grads = backward(X_batch, y_batch, weights)
        
        # Adam optimizer
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads**2
        m_hat = m / (1 - beta1**(epoch+1))
        v_hat = v / (1 - beta2**(epoch+1))
        
        # Update weights
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        epoch_loss += loss
        num_batches += 1
    
    # Validation
    val_loss = validate(X_val, y_val, weights)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_weights(weights)
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    # Print progress
    avg_train_loss = epoch_loss / num_batches
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {val_loss:.4f}")
```

## Key Takeaways

✅ Gradient descent is the foundation of neural network training  
✅ Mini-batch GD provides the best balance of speed and stability  
✅ Adam is the go-to optimizer for most tasks  
✅ Learning rate is the most important hyperparameter  
✅ Monitor both training and validation metrics  
✅ Use regularization to prevent overfitting  
✅ Early stopping saves time and prevents overfitting

## Congratulations! 🎉

You've completed the Neural Networks from Scratch course! You now understand:

- The mathematical foundations (derivatives, functions)
- How neural networks process information (forward propagation)
- How they learn (backpropagation)
- How to train them effectively (optimization)

**Next steps:**
- Implement a neural network from scratch in Python
- Try different architectures (CNN, RNN, Transformer)
- Work on real projects and datasets
- Explore advanced topics (attention mechanisms, GANs, etc.)

Keep learning and building! 🚀

