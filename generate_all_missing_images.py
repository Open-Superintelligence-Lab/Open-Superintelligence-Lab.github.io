import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

BASE_PATH = '/Users/vukrosic/AI Science Projects/open-superintelligence-lab-github-io/public/content/learn/'

# ============================================================================
# NEURON FROM SCRATCH IMAGES
# ============================================================================

def create_linear_step_visual():
    """Linear step: weighted sum visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Linear Step: Weighted Sum', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    # Inputs × Weights = Products
    y_pos = 5.5
    inputs = [2, 3, 1]
    weights = [0.5, -0.3, 0.8]
    products = [1.0, -0.9, 0.8]
    
    for i, (inp, w, prod) in enumerate(zip(inputs, weights, products)):
        y = y_pos - i*1.2
        
        # Input
        box1 = patches.FancyBboxPatch((1, y), 0.8, 0.8, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#3B82F6', linewidth=2)
        ax.add_patch(box1)
        ax.text(1.4, y+0.4, str(inp), fontsize=28, fontweight='bold', color='white', ha='center', va='center')
        
        # ×
        ax.text(2.3, y+0.4, '×', fontsize=32, color='white', ha='center')
        
        # Weight
        box2 = patches.FancyBboxPatch((2.8, y), 0.9, 0.8, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#F59E0B', linewidth=2)
        ax.add_patch(box2)
        ax.text(3.25, y+0.4, str(w), fontsize=26, fontweight='bold', color='white', ha='center', va='center')
        
        # =
        ax.text(4.2, y+0.4, '=', fontsize=32, color='white', ha='center')
        
        # Product
        box3 = patches.FancyBboxPatch((4.7, y), 0.9, 0.8, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#8B5CF6', linewidth=2)
        ax.add_patch(box3)
        ax.text(5.15, y+0.4, str(prod), fontsize=26, fontweight='bold', color='white', ha='center', va='center')
    
    # Sum
    ax.text(6.5, 4, '+', fontsize=40, color='white', ha='center', fontweight='bold')
    
    sum_box = patches.FancyBboxPatch((7.5, 3.5), 2, 1, 
                                      boxstyle="round,pad=0.1", 
                                      edgecolor='white', facecolor='#10B981', linewidth=3)
    ax.add_patch(sum_box)
    ax.text(8.5, 4, '1.9', fontsize=40, fontweight='bold', color='white', ha='center', va='center')
    ax.text(8.5, 3, '+ bias', fontsize=22, color='#94A3B8', ha='center')
    
    # Formula
    ax.text(7, 1.5, 'z = (2×0.5) + (3×-0.3) + (1×0.8) + bias', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    ax.text(7, 0.8, 'z = 1.0 - 0.9 + 0.8 + 0 = 1.9', 
            fontsize=24, color='#10B981', ha='center', fontweight='bold')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/the-linear-step/linear-step-visual.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_activation_comparison():
    """Compare different activations"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Common Activation Functions', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    # Same input for all
    ax.text(7, 6.7, 'Input: [-2, -1, 0, 1, 2]', fontsize=26, color='white', ha='center')
    
    activations = [
        ('ReLU', [0, 0, 0, 1, 2], '#10B981'),
        ('Sigmoid', [0.12, 0.27, 0.50, 0.73, 0.88], '#F59E0B'),
        ('Tanh', [-0.96, -0.76, 0.00, 0.76, 0.96], '#8B5CF6'),
    ]
    
    y_start = 5.5
    for idx, (name, outputs, color) in enumerate(activations):
        y = y_start - idx*1.8
        
        # Name
        ax.text(1.5, y+0.5, name, fontsize=26, color=color, ha='center', fontweight='bold')
        
        # Outputs
        x_start = 3
        for i, val in enumerate(outputs):
            box = patches.FancyBboxPatch((x_start + i*1.3, y), 1, 0.7, 
                                          boxstyle="round,pad=0.05", 
                                          edgecolor='white', facecolor=color, linewidth=2)
            ax.add_patch(box)
            if isinstance(val, float):
                ax.text(x_start + i*1.3 + 0.5, y+0.35, f'{val:.2f}', 
                        fontsize=22, fontweight='bold', color='white', ha='center', va='center')
            else:
                ax.text(x_start + i*1.3 + 0.5, y+0.35, str(val), 
                        fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Note
    ax.text(7, 0.8, 'Different activations, different behaviors!', 
            fontsize=24, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/the-activation-function/activation-comparison.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_loss_visual():
    """Loss function visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create data
    predictions = np.linspace(0, 2, 100)
    target = 1.0
    loss = (predictions - target) ** 2
    
    # Plot
    ax.plot(predictions, loss, 'c-', linewidth=4, label='Loss = (pred - target)²')
    ax.axvline(x=target, color='green', linestyle='--', linewidth=3, label='Target = 1.0')
    ax.plot(target, 0, 'go', markersize=15, label='Minimum loss')
    
    # Mark examples
    ax.plot(0.5, (0.5-1)**2, 'ro', markersize=12)
    ax.text(0.5, (0.5-1)**2 + 0.1, 'Bad prediction', fontsize=20, color='red', ha='center')
    
    ax.plot(0.95, (0.95-1)**2, 'yo', markersize=12)
    ax.text(0.95, (0.95-1)**2 + 0.1, 'Good prediction', fontsize=20, color='yellow', ha='center')
    
    # Labels
    ax.set_xlabel('Prediction', fontsize=28, color='white', fontweight='bold')
    ax.set_ylabel('Loss', fontsize=28, color='white', fontweight='bold')
    ax.set_title('Loss Function: Measures Error', fontsize=36, color='white', fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white', labelsize=20)
    ax.legend(fontsize=20, loc='upper right', facecolor='#334155', edgecolor='white', 
              labelcolor='white', framealpha=0.9)
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/the-concept-of-loss/loss-function.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_learning_process():
    """Learning process: weights adjusting"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Learning: Adjusting Weights to Reduce Loss', 
            fontsize=34, fontweight='bold', color='white', ha='center')
    
    # Timeline
    epochs = ['Start', 'Epoch 10', 'Epoch 50', 'Epoch 100']
    weights = [0.1, 0.4, 0.8, 1.0]
    losses = [2.5, 1.2, 0.3, 0.05]
    colors = ['#EF4444', '#F59E0B', '#10B981', '#10B981']
    
    y_pos = 5.5
    for i, (epoch, w, loss, color) in enumerate(zip(epochs, weights, losses, colors)):
        x = 1.5 + i*3
        
        # Epoch label
        ax.text(x+0.6, 6.5, epoch, fontsize=22, color='white', ha='center', fontweight='bold')
        
        # Weight box
        box1 = patches.FancyBboxPatch((x, y_pos), 1.2, 0.8, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#3B82F6', linewidth=2)
        ax.add_patch(box1)
        ax.text(x+0.6, y_pos+0.4, f'w={w:.1f}', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
        
        # Loss box
        box2 = patches.FancyBboxPatch((x, y_pos-1.2), 1.2, 0.8, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor=color, linewidth=2)
        ax.add_patch(box2)
        ax.text(x+0.6, y_pos-0.8, f'L={loss:.2f}', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
        
        # Arrow
        if i < len(epochs) - 1:
            ax.annotate('', xy=(x+1.8, 5), xytext=(x+1.5, 5),
                        arrowprops=dict(arrowstyle='->', lw=3, color='white'))
    
    # Bottom explanation
    ax.text(7, 2.5, 'Weight gets closer to optimal value (1.0)', 
            fontsize=26, color='#3B82F6', ha='center', fontweight='bold')
    ax.text(7, 1.8, 'Loss decreases from 2.5 → 0.05', 
            fontsize=26, color='#10B981', ha='center', fontweight='bold')
    ax.text(7, 1, 'Learning = Automatic weight adjustment!', 
            fontsize=24, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/the-concept-of-learning/learning-process.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_prediction_flow():
    """Making a prediction flow diagram"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(7, 6.5, 'Forward Pass: Making a Prediction', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    steps = ['Input', 'Linear\n(w·x+b)', 'Activation\n(ReLU)', 'Output']
    values = ['[1, 2]', '0.9', '0.9', '0.9']
    colors = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6']
    
    for i, (step, val, color) in enumerate(zip(steps, values, colors)):
        x = 1 + i*3.5
        
        # Box
        box = patches.FancyBboxPatch((x, 3.5), 2, 1.5, 
                                      boxstyle="round,pad=0.1", 
                                      edgecolor='white', facecolor=color, linewidth=3)
        ax.add_patch(box)
        
        # Step name
        ax.text(x+1, 5.3, step, fontsize=22, fontweight='bold', color='white', ha='center', va='top')
        
        # Value
        ax.text(x+1, 4, val, fontsize=28, fontweight='bold', color='white', ha='center', va='center')
        
        # Arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+2.5, 4.25), xytext=(x+2.2, 4.25),
                        arrowprops=dict(arrowstyle='->', lw=4, color='white'))
    
    # Bottom note
    ax.text(7, 2, 'Data flows forward through the network', 
            fontsize=26, color='#94A3B8', ha='center', style='italic')
    ax.text(7, 1.3, 'Input → Transform → Activate → Prediction', 
            fontsize=24, color='#94A3B8', ha='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/making-a-prediction/prediction-flow.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_neuron_code_visual():
    """Building a neuron code visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Neuron Components in Code', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    components = [
        ('nn.Linear()', 'Weights & Bias', '#3B82F6'),
        ('nn.ReLU()', 'Activation', '#F59E0B'),
        ('forward()', 'Computation', '#10B981'),
        ('backward()', 'Learning', '#8B5CF6'),
    ]
    
    y_start = 6
    for i, (code, desc, color) in enumerate(components):
        y = y_start - i*1.4
        
        # Code box
        box = patches.FancyBboxPatch((2, y), 4, 0.9, 
                                      boxstyle="round,pad=0.1", 
                                      edgecolor='white', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(4, y+0.45, code, fontsize=26, fontweight='bold', color='white', ha='center', va='center',
                family='monospace')
        
        # Description
        ax.text(7, y+0.45, '→', fontsize=32, color='white', ha='center')
        ax.text(9.5, y+0.45, desc, fontsize=24, color='white', ha='left')
    
    # Bottom note
    ax.text(7, 0.8, 'PyTorch handles all the complexity!', 
            fontsize=26, color='#94A3B8', ha='center', fontweight='bold')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neuron-from-scratch/building-a-neuron-in-python/neuron-code.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# ============================================================================
# NEURAL NETWORKS IMAGES
# ============================================================================

def create_network_layers():
    """Network architecture layers visualization"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Neural Network Architecture', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    layers = [
        ('Input\nLayer', 784, '#3B82F6', 1.5),
        ('Hidden\nLayer 1', 128, '#10B981', 4.5),
        ('Hidden\nLayer 2', 64, '#F59E0B', 7.5),
        ('Output\nLayer', 10, '#8B5CF6', 10.5),
    ]
    
    for name, size, color, x in layers:
        # Draw neurons
        num_display = min(size, 8)
        y_start = 6 - (num_display * 0.4)
        
        for i in range(num_display):
            y = y_start + i*0.8
            circle = plt.Circle((x, y), 0.25, color=color, ec='white', linewidth=2)
            ax.add_patch(circle)
            
            if i == num_display - 1 and size > num_display:
                ax.text(x, y-0.6, '...', fontsize=24, color=color, ha='center')
        
        # Label
        ax.text(x, 7.5, name, fontsize=22, color='white', ha='center', fontweight='bold')
        ax.text(x, 2, f'{size}', fontsize=20, color='#94A3B8', ha='center')
        
        # Connections
        if x < 10:
            for i in range(min(3, num_display)):
                for j in range(min(3, num_display)):
                    y1 = y_start + i*0.8
                    y2 = y_start + j*0.8
                    ax.plot([x+0.25, x+3-0.25], [y1, y2], 'white', alpha=0.2, linewidth=1)
    
    # Bottom note
    ax.text(7, 1, 'Each layer transforms data: 784 → 128 → 64 → 10', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neural-networks/architecture-of-a-network/network-layers.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_layer_structure():
    """Single layer structure"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(7, 6.5, 'Layer = Multiple Neurons in Parallel', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Input
    ax.text(2, 5.5, 'Input (3)', fontsize=24, color='white', ha='center')
    for i in range(3):
        circle = plt.Circle((2, 4.5-i*0.8), 0.3, color='#3B82F6', ec='white', linewidth=2)
        ax.add_patch(circle)
    
    # Neurons in layer
    ax.text(7, 5.5, 'Layer (4 neurons)', fontsize=24, color='white', ha='center')
    for i in range(4):
        circle = plt.Circle((7, 5-i), 0.35, color='#10B981', ec='white', linewidth=3)
        ax.add_patch(circle)
        
        # Connections from all inputs
        for j in range(3):
            ax.plot([2.3, 6.65], [4.5-j*0.8, 5-i], 'white', alpha=0.3, linewidth=1.5)
    
    # Output
    ax.text(12, 5.5, 'Output (4)', fontsize=24, color='white', ha='center')
    for i in range(4):
        circle = plt.Circle((12, 5-i), 0.3, color='#8B5CF6', ec='white', linewidth=2)
        ax.add_patch(circle)
        ax.plot([7.35, 11.7], [5-i, 5-i], 'white', alpha=0.4, linewidth=2)
    
    # Note
    ax.text(7, 1.5, 'Each neuron receives ALL inputs', 
            fontsize=26, color='#94A3B8', ha='center', fontweight='bold')
    ax.text(7, 0.8, 'nn.Linear(3, 4) creates this layer', 
            fontsize=24, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'neural-networks/building-a-layer/layer-structure.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# Create all neuron and network images
print("Creating neuron-from-scratch images...")
create_linear_step_visual()
create_activation_comparison()
create_loss_visual()
create_learning_process()
create_prediction_flow()
create_neuron_code_visual()

print("Creating neural-networks images...")
create_network_layers()
create_layer_structure()

print("Part 1 complete! Run generate_all_missing_images_part2.py for attention/transformer images...")

