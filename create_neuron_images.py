import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

def create_biological_neuron():
    """Create biological vs artificial neuron comparison"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Biological vs Artificial Neuron', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    # Biological side
    ax.text(3.5, 7.7, 'Biological Neuron', fontsize=28, color='#10B981', ha='center', fontweight='bold')
    
    # Dendrites (inputs)
    for i in range(3):
        y = 6.5 - i*0.8
        ax.plot([0.5, 2], [y, 6], 'c-', linewidth=3)
        ax.text(0.3, y, f'Input {i+1}', fontsize=18, color='#94A3B8', ha='right')
    
    # Cell body
    cell = plt.Circle((2.5, 6), 0.7, color='#10B981', ec='white', linewidth=3)
    ax.add_patch(cell)
    ax.text(2.5, 6, 'Cell', fontsize=20, fontweight='bold', color='white', ha='center')
    
    # Axon (output)
    ax.plot([3.2, 5.5], [6, 6], 'c-', linewidth=4)
    ax.text(5.7, 6, 'Output', fontsize=18, color='#94A3B8', ha='left')
    
    # Artificial side
    ax.text(10.5, 7.7, 'Artificial Neuron', fontsize=28, color='#F59E0B', ha='center', fontweight='bold')
    
    # Inputs with weights
    inputs = ['x₁', 'x₂', 'x₃']
    weights = ['w₁', 'w₂', 'w₃']
    
    for i in range(3):
        y = 6.5 - i*0.8
        ax.plot([7.5, 9], [y, 6], color='#F59E0B', linewidth=3)
        ax.text(7.3, y, inputs[i], fontsize=22, color='white', ha='right', fontweight='bold')
        ax.text(8.2, y - 0.2, weights[i], fontsize=16, color='#94A3B8', ha='center')
    
    # Neuron circle
    neuron = plt.Circle((9.5, 6), 0.7, color='#F59E0B', ec='white', linewidth=3)
    ax.add_patch(neuron)
    ax.text(9.5, 6, 'Σ', fontsize=32, fontweight='bold', color='white', ha='center')
    
    # Output
    ax.plot([10.2, 12], [6, 6], color='#F59E0B', linewidth=4)
    ax.text(12.2, 6, 'y', fontsize=22, color='white', ha='left', fontweight='bold')
    
    # Formula below
    ax.text(10.5, 4.8, 'y = f(Σ wᵢxᵢ + b)', fontsize=26, color='#F59E0B', ha='center', fontweight='bold')
    
    # Bottom explanation
    ax.text(7, 3.5, 'Both process multiple inputs into single output!', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    
    # Key differences
    ax.text(3.5, 2.5, 'Biological:', fontsize=22, color='#10B981', ha='center', fontweight='bold')
    ax.text(3.5, 2, '• Electrochemical', fontsize=18, color='#94A3B8', ha='center')
    ax.text(3.5, 1.6, '• Slow (milliseconds)', fontsize=18, color='#94A3B8', ha='center')
    ax.text(3.5, 1.2, '• Complex', fontsize=18, color='#94A3B8', ha='center')
    
    ax.text(10.5, 2.5, 'Artificial:', fontsize=22, color='#F59E0B', ha='center', fontweight='bold')
    ax.text(10.5, 2, '• Mathematical', fontsize=18, color='#94A3B8', ha='center')
    ax.text(10.5, 1.6, '• Fast (nanoseconds)', fontsize=18, color='#94A3B8', ha='center')
    ax.text(10.5, 1.2, '• Simple', fontsize=18, color='#94A3B8', ha='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig('/Users/vukrosic/AI Science Projects/open-superintelligence-lab-github-io/public/content/learn/neuron-from-scratch/what-is-a-neuron/biological-vs-artificial.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_neuron_parts():
    """Create neuron components diagram"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Parts of an Artificial Neuron', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    # Inputs
    ax.text(2, 7.5, '1. Inputs', fontsize=26, color='#10B981', ha='center', fontweight='bold')
    
    inputs = [2, 3, 1]
    y_start = 6.5
    
    for i, val in enumerate(inputs):
        box = patches.FancyBboxPatch((0.8, y_start - i*0.9), 0.8, 0.7, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor='#10B981', linewidth=2)
        ax.add_patch(box)
        ax.text(1.2, y_start - i*0.9 + 0.35, str(val), 
                fontsize=28, fontweight='bold', color='white', ha='center', va='center')
        ax.text(2.2, y_start - i*0.9 + 0.35, f'x{i+1}', 
                fontsize=22, color='white', ha='left')
    
    # Weights
    ax.text(4.5, 7.5, '2. Weights', fontsize=26, color='#F59E0B', ha='center', fontweight='bold')
    
    weights = [0.5, -0.3, 0.8]
    
    for i, val in enumerate(weights):
        box = patches.FancyBboxPatch((3.8, y_start - i*0.9), 0.9, 0.7, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor='#F59E0B', linewidth=2)
        ax.add_patch(box)
        ax.text(4.25, y_start - i*0.9 + 0.35, str(val), 
                fontsize=26, fontweight='bold', color='white', ha='center', va='center')
        ax.text(5.2, y_start - i*0.9 + 0.35, f'w{i+1}', 
                fontsize=22, color='white', ha='left')
    
    # Multiply
    ax.text(6.5, 5.5, '×', fontsize=36, color='white', ha='center')
    
    # Products
    ax.text(8, 7.5, '3. Products', fontsize=26, color='#6366F1', ha='center', fontweight='bold')
    
    products = [1.0, -0.9, 0.8]
    
    for i, val in enumerate(products):
        box = patches.FancyBboxPatch((7.2, y_start - i*0.9), 1, 0.7, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor='#6366F1', linewidth=2)
        ax.add_patch(box)
        ax.text(7.7, y_start - i*0.9 + 0.35, str(val), 
                fontsize=26, fontweight='bold', color='white', ha='center', va='center')
    
    # Sum
    ax.text(9.5, 5.5, '→', fontsize=36, color='white', ha='center')
    
    ax.text(11, 7.5, '4. Sum', fontsize=26, color='#8B5CF6', ha='center', fontweight='bold')
    
    sum_box = patches.FancyBboxPatch((10, 5.8), 2, 0.9, 
                                      boxstyle="round,pad=0.1", 
                                      edgecolor='white', facecolor='#8B5CF6', linewidth=3)
    ax.add_patch(sum_box)
    ax.text(11, 6.25, '0.9', 
            fontsize=36, fontweight='bold', color='white', ha='center', va='center')
    
    ax.text(11, 5.1, '+ bias', fontsize=20, color='#94A3B8', ha='center')
    
    # Activation
    ax.annotate('', xy=(11, 4), xytext=(11, 4.9),
                arrowprops=dict(arrowstyle='->', lw=4, color='#F59E0B'))
    
    ax.text(11, 3.5, '5. Activation', fontsize=26, color='#F59E0B', ha='center', fontweight='bold')
    
    activation_box = patches.FancyBboxPatch((10, 2.2), 2, 0.9, 
                                             boxstyle="round,pad=0.1", 
                                             edgecolor='white', facecolor='#10B981', linewidth=3)
    ax.add_patch(activation_box)
    ax.text(11, 2.65, '0.71', 
            fontsize=36, fontweight='bold', color='white', ha='center', va='center')
    
    ax.text(11, 1.5, 'Final Output', fontsize=22, color='#94A3B8', ha='center', style='italic')
    
    # Formula at bottom
    ax.text(7, 0.5, 'Output = Activation( Σ(weights × inputs) + bias )', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig('/Users/vukrosic/AI Science Projects/open-superintelligence-lab-github-io/public/content/learn/neuron-from-scratch/what-is-a-neuron/neuron-parts.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_simple_neuron_diagram():
    """Create simple single neuron diagram"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(7, 6.5, 'Simple Neuron Example', 
            fontsize=38, fontweight='bold', color='white', ha='center')
    
    # Inputs
    inputs = [(2, '#3B82F6'), (3, '#3B82F6'), (1, '#3B82F6')]
    weights = [(0.5, '#F59E0B'), (-0.3, '#F59E0B'), (0.8, '#F59E0B')]
    
    for i, ((inp, color_i), (w, color_w)) in enumerate(zip(inputs, weights)):
        y = 5 - i
        
        # Input box
        inp_box = patches.FancyBboxPatch((0.5, y - 0.3), 0.8, 0.6, 
                                          boxstyle="round,pad=0.05", 
                                          edgecolor='white', facecolor=color_i, linewidth=2)
        ax.add_patch(inp_box)
        ax.text(0.9, y, str(inp), 
                fontsize=28, fontweight='bold', color='white', ha='center', va='center')
        
        # Connection line with weight
        ax.plot([1.5, 5.5], [y, 3.5], color=color_w, linewidth=3)
        ax.text(2.5, y - 0.3, f'w={w}', 
                fontsize=20, color=color_w, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1E293B', edgecolor=color_w, linewidth=2))
    
    # Neuron circle
    neuron = plt.Circle((6, 3.5), 1.2, color='#8B5CF6', ec='white', linewidth=4)
    ax.add_patch(neuron)
    ax.text(6, 3.8, 'Σ', fontsize=44, fontweight='bold', color='white', ha='center')
    ax.text(6, 3.2, '+b', fontsize=24, color='white', ha='center')
    
    # Calculation
    ax.text(6, 1.8, '(2×0.5) + (3×-0.3) + (1×0.8) + bias', 
            fontsize=22, color='#94A3B8', ha='center')
    ax.text(6, 1.3, '= 1.0 - 0.9 + 0.8 + 0 = 0.9', 
            fontsize=22, color='#F59E0B', ha='center', fontweight='bold')
    
    # Activation and output
    ax.plot([7.2, 9], [3.5, 3.5], color='#10B981', linewidth=4)
    ax.text(8, 4, 'ReLU', fontsize=20, color='#10B981', ha='center', fontweight='bold')
    
    # Output box
    output_box = patches.FancyBboxPatch((9.5, 3.2), 1.2, 0.6, 
                                         boxstyle="round,pad=0.05", 
                                         edgecolor='white', facecolor='#10B981', linewidth=3)
    ax.add_patch(output_box)
    ax.text(10.1, 3.5, '0.9', 
            fontsize=28, fontweight='bold', color='white', ha='center', va='center')
    
    ax.text(10.1, 2.7, 'Output', fontsize=20, color='#94A3B8', ha='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig('/Users/vukrosic/AI Science Projects/open-superintelligence-lab-github-io/public/content/learn/neuron-from-scratch/what-is-a-neuron/simple-neuron.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# Create all images
print("Creating biological vs artificial neuron...")
create_biological_neuron()

print("Creating neuron parts diagram...")
create_neuron_parts()

print("Creating simple neuron diagram...")
create_simple_neuron_diagram()

print("All neuron images created successfully!")

