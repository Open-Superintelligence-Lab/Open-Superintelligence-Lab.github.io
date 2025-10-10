import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

BASE_PATH = '/Users/vukrosic/AI Science Projects/open-superintelligence-lab-github-io/public/content/learn/'

# ============================================================================
# ATTENTION MECHANISM IMAGES
# ============================================================================

def create_attention_concept():
    """What is attention: concept visualization"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Attention: Focus on Relevant Parts', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Sentence
    sentence = "The cat sat on the mat"
    words = sentence.split()
    
    # Word boxes with attention highlights
    ax.text(7, 7.5, 'Query: "What did the cat do?"', fontsize=26, color='#F59E0B', ha='center', fontweight='bold')
    
    # Attention weights
    attention = [0.1, 0.6, 0.2, 0.05, 0.02, 0.03]  # "cat" and "sat" most important
    
    x_start = 2
    y_pos = 5.5
    
    for i, (word, attn) in enumerate(zip(words, attention)):
        alpha = 0.3 + attn * 0.7  # Scale alpha by attention
        size = 1 + attn * 1.5
        color_intensity = int(255 * attn)
        
        # Box with size based on attention
        box = patches.FancyBboxPatch((x_start + i*1.8, y_pos), 1.5, 0.8+attn, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor='#10B981' if attn > 0.3 else '#3B82F6', 
                                      linewidth=2+attn*4)
        ax.add_patch(box)
        ax.text(x_start + i*1.8 + 0.75, y_pos + 0.4 + attn/2, word, 
                fontsize=18+attn*20, fontweight='bold', color='white', ha='center', va='center')
        
        # Attention weight below
        ax.text(x_start + i*1.8 + 0.75, y_pos - 0.4, f'{attn:.0%}', 
                fontsize=18, color='#94A3B8', ha='center')
    
    # Explanation
    ax.text(7, 3.5, '"cat" (60%) and "sat" (20%) are most relevant', 
            fontsize=28, color='#10B981', ha='center', fontweight='bold')
    ax.text(7, 2.8, 'Other words get less attention', 
            fontsize=24, color='#94A3B8', ha='center')
    
    ax.text(7, 1.5, 'Attention weights sum to 100%', 
            fontsize=24, color='#94A3B8', ha='center', style='italic')
    ax.text(7, 0.8, 'Model learns which words to focus on!', 
            fontsize=22, color='#94A3B8', ha='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'attention-mechanism/what-is-attention/attention-concept.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_qkv_visual():
    """Query, Key, Value visualization"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Query, Key, Value Mechanism', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Input
    input_box = patches.FancyBboxPatch((6, 7.5), 2, 0.6, 
                                        boxstyle="round,pad=0.05", 
                                        edgecolor='white', facecolor='#94A3B8', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 7.8, 'Input', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Split to Q, K, V
    components = [
        ('Query', 'What am I\nlooking for?', '#10B981', 2),
        ('Key', 'What do I\ncontain?', '#F59E0B', 7),
        ('Value', 'What info\ndo I have?', '#8B5CF6', 12),
    ]
    
    for name, desc, color, x in components:
        # Arrow from input
        ax.annotate('', xy=(x+0.5, 6.2), xytext=(7, 7.3),
                    arrowprops=dict(arrowstyle='->', lw=3, color=color))
        
        # Component box
        box = patches.FancyBboxPatch((x, 4.8), 2, 1.2, 
                                      boxstyle="round,pad=0.1", 
                                      edgecolor='white', facecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x+1, 5.8, name, fontsize=26, fontweight='bold', color='white', ha='center', va='center')
        ax.text(x+1, 5.2, desc, fontsize=18, color='white', ha='center', va='center')
    
    # Attention computation
    ax.text(7, 3.5, '1. Q × K → Scores', fontsize=24, color='#94A3B8', ha='center')
    ax.text(7, 3, '2. Softmax → Weights', fontsize=24, color='#94A3B8', ha='center')
    ax.text(7, 2.5, '3. Weights × V → Output', fontsize=24, color='#94A3B8', ha='center')
    
    # Output
    output_box = patches.FancyBboxPatch((5.5, 1), 3, 0.8, 
                                         boxstyle="round,pad=0.1", 
                                         edgecolor='white', facecolor='#3B82F6', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7, 1.4, 'Attention Output', fontsize=26, fontweight='bold', color='white', ha='center', va='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'attention-mechanism/what-is-attention/qkv-mechanism.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_attention_scores_matrix():
    """Attention scores matrix visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Attention Score Matrix', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Create attention matrix visualization
    size = 5
    scores = np.random.rand(size, size)
    scores = scores / scores.sum(axis=1, keepdims=True)  # Normalize rows
    
    box_size = 1
    x_start = 2.5
    y_start = 7.5
    
    # Row labels (Query positions)
    for i in range(size):
        ax.text(x_start - 0.7, y_start - i*1.1 + 0.5, f'Q{i}', 
                fontsize=20, color='#10B981', ha='center', fontweight='bold')
    
    # Column labels (Key positions)
    for j in range(size):
        ax.text(x_start + j*1.1 + 0.5, y_start + 0.7, f'K{j}', 
                fontsize=20, color='#F59E0B', ha='center', fontweight='bold')
    
    # Draw matrix
    for i in range(size):
        for j in range(size):
            val = scores[i, j]
            color_intensity = val
            color = plt.cm.viridis(color_intensity)
            
            rect = patches.FancyBboxPatch((x_start + j*1.1, y_start - i*1.1), box_size, box_size, 
                                           boxstyle="round,pad=0.05", 
                                           edgecolor='white', facecolor=color, linewidth=1)
            ax.add_patch(rect)
            ax.text(x_start + j*1.1 + 0.5, y_start - i*1.1 + 0.5, f'{val:.2f}', 
                    fontsize=16, fontweight='bold', color='white', ha='center', va='center')
    
    # Note
    ax.text(6, 1.5, 'Each row shows where one position attends', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    ax.text(6, 0.9, 'Darker = Higher attention', 
            fontsize=22, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'attention-mechanism/calculating-attention-scores/attention-matrix.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_multi_head_visualization():
    """Multi-head attention visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Multi-Head Attention: 8 Heads in Parallel', 
            fontsize=34, fontweight='bold', color='white', ha='center')
    
    # Input
    input_box = patches.FancyBboxPatch((6, 6.5), 2, 0.6, 
                                        boxstyle="round,pad=0.05", 
                                        edgecolor='white', facecolor='#3B82F6', linewidth=3)
    ax.add_patch(input_box)
    ax.text(7, 6.8, 'Input', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # 8 heads
    num_heads = 8
    colors = plt.cm.tab10(np.linspace(0, 1, num_heads))
    
    y_start = 5
    for i in range(num_heads):
        x = 1.5 + i*1.5
        
        # Arrow from input
        ax.plot([7, x+0.4], [6.4, y_start+0.6], 'white', alpha=0.3, linewidth=2)
        
        # Head box
        box = patches.FancyBboxPatch((x, y_start - 0.3), 0.8, 0.6, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor=colors[i], linewidth=2)
        ax.add_patch(box)
        ax.text(x+0.4, y_start, f'H{i+1}', fontsize=18, fontweight='bold', color='white', ha='center', va='center')
        
        # Arrow to concat
        ax.plot([x+0.4, 7], [y_start-0.5, 3.2], 'white', alpha=0.3, linewidth=2)
    
    # Concatenate
    concat_box = patches.FancyBboxPatch((5, 2.5), 4, 0.6, 
                                         boxstyle="round,pad=0.05", 
                                         edgecolor='white', facecolor='#F59E0B', linewidth=3)
    ax.add_patch(concat_box)
    ax.text(7, 2.8, 'Concatenate Heads', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Output projection
    ax.annotate('', xy=(7, 1.5), xytext=(7, 2.3),
                arrowprops=dict(arrowstyle='->', lw=4, color='white'))
    
    output_box = patches.FancyBboxPatch((6, 0.5), 2, 0.8, 
                                         boxstyle="round,pad=0.1", 
                                         edgecolor='white', facecolor='#8B5CF6', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7, 0.9, 'Output', fontsize=26, fontweight='bold', color='white', ha='center', va='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'attention-mechanism/multi-head-attention/multi-head-visual.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_self_attention_visual():
    """Self-attention concept"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Self-Attention: Sequence Attends to Itself', 
            fontsize=34, fontweight='bold', color='white', ha='center')
    
    words = ['The', 'cat', 'sat']
    positions = [3, 7, 11]
    
    for i, (word, x) in enumerate(zip(words, positions)):
        # Word box
        box = patches.FancyBboxPatch((x-0.8, 5.5), 1.6, 0.8, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white', facecolor='#3B82F6', linewidth=3)
        ax.add_patch(box)
        ax.text(x, 5.9, word, fontsize=28, fontweight='bold', color='white', ha='center', va='center')
        
        # Show attention connections
        for j, (word2, x2) in enumerate(zip(words, positions)):
            if i != j:
                # Attention line
                alpha = 0.5 if abs(i-j) == 1 else 0.2
                ax.plot([x, x2], [5.3, 5.3], 'c-', alpha=alpha, linewidth=2+alpha*4)
                ax.plot([x, x2], [5.3, 5.3], 'co', markersize=8, alpha=alpha)
    
    # Explanation
    ax.text(7, 3.8, 'Each word attends to ALL words (including itself)', 
            fontsize=26, color='#94A3B8', ha='center', fontweight='bold')
    ax.text(7, 3.2, '"cat" learns from "The" and "sat" for context', 
            fontsize=24, color='#10B981', ha='center')
    ax.text(7, 2.6, 'Q, K, V all come from the same sequence!', 
            fontsize=22, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'attention-mechanism/self-attention-from-scratch/self-attention-concept.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# ============================================================================
# TRANSFORMER IMAGES
# ============================================================================

def create_transformer_architecture_diagram():
    """Full transformer architecture"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(6, 13.5, 'Transformer Architecture', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Input
    box = patches.FancyBboxPatch((4, 12.5), 4, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#3B82F6', linewidth=2)
    ax.add_patch(box)
    ax.text(6, 12.85, 'Input Tokens', fontsize=22, fontweight='bold', color='white', ha='center', va='center')
    
    # Embeddings
    y = 11.5
    box = patches.FancyBboxPatch((4, y), 4, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#6366F1', linewidth=2)
    ax.add_patch(box)
    ax.text(6, y+0.35, 'Embeddings + Positions', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
    ax.plot([6, 6], [y+0.8, y+1.4], 'white', linewidth=3)
    
    # Transformer blocks (N times)
    for block_idx in range(3):
        y_block = 10 - block_idx*3
        
        # Block container
        block_box = patches.FancyBboxPatch((3, y_block-2.5), 6, 2.3, 
                                            boxstyle="round,pad=0.1", 
                                            edgecolor='cyan', facecolor='#1E293B', linewidth=2, linestyle='--')
        ax.add_patch(block_box)
        ax.text(9.2, y_block - 1.3, f'Block {block_idx+1}', fontsize=18, color='cyan', ha='left')
        
        # Multi-head attention
        box1 = patches.FancyBboxPatch((4, y_block-0.7), 4, 0.6, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#10B981', linewidth=2)
        ax.add_patch(box1)
        ax.text(6, y_block-0.4, 'Multi-Head Attention', fontsize=18, fontweight='bold', color='white', ha='center', va='center')
        
        # FFN
        box2 = patches.FancyBboxPatch((4, y_block-1.9), 4, 0.6, 
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='white', facecolor='#F59E0B', linewidth=2)
        ax.add_patch(box2)
        ax.text(6, y_block-1.6, 'Feed-Forward', fontsize=18, fontweight='bold', color='white', ha='center', va='center')
        
        # Arrows
        ax.plot([6, 6], [y_block-0.1, y_block-1.3], 'white', linewidth=2)
        
        if block_idx < 2:
            ax.plot([6, 6], [y_block-2.6, y_block-3.2], 'white', linewidth=2)
    
    # Output head
    y_out = 1
    box = patches.FancyBboxPatch((4, y_out), 4, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#8B5CF6', linewidth=2)
    ax.add_patch(box)
    ax.text(6, y_out+0.35, 'Output Projection', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'building-a-transformer/transformer-architecture/transformer-diagram.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_transformer_block_diagram():
    """Transformer block internal structure"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Transformer Block Components', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Input
    y = 8.5
    box = patches.FancyBboxPatch((5.5, y), 3, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#3B82F6', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.35, 'Input', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Attention sub-block
    y = 7
    ax.text(3, y+1, '1. Attention Sub-block', fontsize=22, color='#10B981', ha='left', fontweight='bold')
    
    box = patches.FancyBboxPatch((4, y), 6, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#10B981', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.3, 'Multi-Head Attention', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
    
    # Add & Norm
    y = 6
    box = patches.FancyBboxPatch((4.5, y), 5, 0.5, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#6366F1', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.25, 'Add & Norm (Residual)', fontsize=18, color='white', ha='center', va='center')
    
    # FFN sub-block
    y = 4.8
    ax.text(3, y+1, '2. FFN Sub-block', fontsize=22, color='#F59E0B', ha='left', fontweight='bold')
    
    box = patches.FancyBboxPatch((4, y), 6, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#F59E0B', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.3, 'Feed-Forward Network', fontsize=20, fontweight='bold', color='white', ha='center', va='center')
    
    # Add & Norm
    y = 3.8
    box = patches.FancyBboxPatch((4.5, y), 5, 0.5, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#6366F1', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.25, 'Add & Norm (Residual)', fontsize=18, color='white', ha='center', va='center')
    
    # Output
    y = 2.5
    box = patches.FancyBboxPatch((5.5, y), 3, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='white', facecolor='#8B5CF6', linewidth=2)
    ax.add_patch(box)
    ax.text(7, y+0.35, 'Output', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Note
    ax.text(7, 1.2, 'Attention → Add&Norm → FFN → Add&Norm', 
            fontsize=24, color='#94A3B8', ha='center', fontweight='bold')
    ax.text(7, 0.5, 'Residual connections help gradients flow!', 
            fontsize=22, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'building-a-transformer/building-a-transformer-block/block-diagram.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# ============================================================================
# MOE IMAGES
# ============================================================================

def create_moe_routing():
    """MoE routing visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Mixture of Experts: Sparse Routing', 
            fontsize=36, fontweight='bold', color='white', ha='center')
    
    # Input token
    token_box = patches.FancyBboxPatch((6, 8.5), 2, 0.7, 
                                        boxstyle="round,pad=0.05", 
                                        edgecolor='white', facecolor='#3B82F6', linewidth=3)
    ax.add_patch(token_box)
    ax.text(7, 8.85, 'Token', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    # Router
    ax.annotate('', xy=(7, 7.5), xytext=(7, 8.3),
                arrowprops=dict(arrowstyle='->', lw=3, color='white'))
    
    router_box = patches.FancyBboxPatch((5.5, 6.8), 3, 0.6, 
                                         boxstyle="round,pad=0.05", 
                                         edgecolor='white', facecolor='#F59E0B', linewidth=2)
    ax.add_patch(router_box)
    ax.text(7, 7.1, 'Router', fontsize=22, fontweight='bold', color='white', ha='center', va='center')
    
    # 8 Experts
    num_experts = 8
    expert_colors = ['#10B981', '#EF4444', '#94A3B8', '#94A3B8', '#94A3B8', '#10B981', '#94A3B8', '#94A3B8']
    active = [True, False, False, False, False, True, False, False]
    
    y_experts = 5
    for i in range(num_experts):
        x = 1.5 + i*1.5
        
        # Expert box
        box = patches.FancyBboxPatch((x, y_experts), 0.9, 0.7, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='white' if active[i] else '#4B5563', 
                                      facecolor=expert_colors[i], 
                                      linewidth=3 if active[i] else 1,
                                      alpha=1.0 if active[i] else 0.3)
        ax.add_patch(box)
        ax.text(x+0.45, y_experts+0.35, f'E{i}', fontsize=18, fontweight='bold', color='white', ha='center', va='center')
        
        # Connection from router
        alpha = 1.0 if active[i] else 0.15
        linewidth = 3 if active[i] else 1
        ax.plot([7, x+0.45], [6.7, y_experts+0.8], color=expert_colors[i] if active[i] else '#4B5563', 
                alpha=alpha, linewidth=linewidth)
    
    # Output
    ax.text(7, 3.5, 'Top-2 Experts Selected: E0 (60%) + E5 (40%)', 
            fontsize=26, color='#10B981', ha='center', fontweight='bold')
    
    output_box = patches.FancyBboxPatch((5, 2.3), 4, 0.8, 
                                         boxstyle="round,pad=0.1", 
                                         edgecolor='white', facecolor='#8B5CF6', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7, 2.7, 'Combined Output', fontsize=24, fontweight='bold', color='white', ha='center', va='center')
    
    ax.text(7, 1, 'Only 2 of 8 experts activated (sparse!)', 
            fontsize=24, color='#94A3B8', ha='center', style='italic')
    
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'transformer-feedforward/what-is-mixture-of-experts/moe-routing.png', 
                dpi=150, facecolor='#1E293B', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# Create all images
print("Creating attention mechanism images...")
create_attention_concept()
create_qkv_visual()
create_attention_scores_matrix()
create_multi_head_visualization()
create_self_attention_visual()

print("Creating transformer images...")
create_transformer_architecture_diagram()
create_transformer_block_diagram()

print("Creating MoE images...")
create_moe_routing()

print("\n✅ All missing images created successfully!")

