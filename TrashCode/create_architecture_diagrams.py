"""
Architecture Visualization for Physics-Informed Transformer

This creates a visual diagram of the training pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_architecture_diagram():
    """Create a visual diagram of the transformer architecture."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Physics-Informed Transformer Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 11, 'Gradients flow through entire pipeline ✓', 
            fontsize=12, ha='center', style='italic', color='green')
    
    # Define colors
    color_data = '#E8F4F8'
    color_model = '#FFF4E6'
    color_physics = '#F0E6FF'
    color_loss = '#FFE6E6'
    
    # Layer 1: Input Data
    y_pos = 9.5
    input_box = FancyBboxPatch((2, y_pos), 6, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_data, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, y_pos + 0.4, 'Input: Kinematic Data', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, y_pos + 0.1, 'qpos, qvel, qacc [batch, seq_len, 3×nv]', 
            fontsize=9, ha='center', family='monospace')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.7),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2, color='blue')
    ax.add_patch(arrow1)
    
    # Layer 2: Embedding
    y_pos -= 1.5
    embed_box = FancyBboxPatch((2, y_pos), 6, 0.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(embed_box)
    ax.text(5, y_pos + 0.4, 'Linear Projection + Positional Encoding', 
            fontsize=11, fontweight='bold', ha='center')
    ax.text(5, y_pos + 0.1, '[batch, seq_len, d_model=256]', 
            fontsize=9, ha='center', family='monospace')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.7),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color='blue')
    ax.add_patch(arrow2)
    
    # Layer 3: Transformer Blocks
    y_pos -= 1.5
    for i in range(3):
        block_y = y_pos - i * 0.35
        trans_box = FancyBboxPatch((2.2, block_y), 5.6, 0.3,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='darkblue', facecolor=color_model, 
                                  linewidth=1.5, alpha=0.8)
        ax.add_patch(trans_box)
        ax.text(5, block_y + 0.15, f'Transformer Block {i+1}: Multi-Head Attention + FFN', 
                fontsize=8, ha='center')
    
    ax.text(1, y_pos - 0.5, f'×6 layers', fontsize=9, ha='center', style='italic')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((5, y_pos - 1.2), (5, y_pos - 1.9),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color='blue')
    ax.add_patch(arrow3)
    
    # Layer 4: Output Projection
    y_pos -= 2.7
    output_box = FancyBboxPatch((2, y_pos), 6, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, y_pos + 0.4, 'Output: GRF & COP Predictions', 
            fontsize=11, fontweight='bold', ha='center')
    ax.text(5, y_pos + 0.1, '[batch, seq_len, 12] = 2×(3 GRF + 3 COP)', 
            fontsize=9, ha='center', family='monospace')
    
    # Arrow 4 (split)
    arrow4 = FancyArrowPatch((5, y_pos), (3.5, y_pos - 0.7),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color='green')
    ax.add_patch(arrow4)
    
    # Add Jacobian input
    y_pos_jac = y_pos - 0.8
    jac_box = FancyBboxPatch((6.5, y_pos_jac), 2.5, 0.6,
                            boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='white', 
                            linewidth=2, linestyle='--')
    ax.add_patch(jac_box)
    ax.text(7.75, y_pos_jac + 0.3, 'Jacobian J', 
            fontsize=10, fontweight='bold', ha='center', color='purple')
    ax.text(7.75, y_pos_jac + 0.05, '[seq_len, nv, 12]', 
            fontsize=8, ha='center', family='monospace')
    
    # Arrow from Jacobian
    arrow_jac = FancyArrowPatch((6.5, y_pos_jac + 0.3), (4.5, y_pos_jac),
                               arrowstyle='->', mutation_scale=25,
                               linewidth=2, color='purple', linestyle='--')
    ax.add_patch(arrow_jac)
    
    # Layer 5: Physics Layer (Jacobian multiplication)
    y_pos -= 1.5
    physics_box = FancyBboxPatch((2, y_pos), 6, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='purple', facecolor=color_physics, linewidth=3)
    ax.add_patch(physics_box)
    ax.text(5, y_pos + 0.5, '⚡ DIFFERENTIABLE PHYSICS LAYER ⚡', 
            fontsize=11, fontweight='bold', ha='center', color='purple')
    ax.text(5, y_pos + 0.15, 'τ = J^T @ [GRF_left, GRF_right, COP_left, COP_right]', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, y_pos - 0.15, '[batch, seq_len, nv] joint torques', 
            fontsize=8, ha='center', family='monospace')
    
    # Arrow 5
    arrow5 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.7),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color='red')
    ax.add_patch(arrow5)
    
    # Layer 6: Loss Computation
    y_pos -= 1.5
    loss_box = FancyBboxPatch((2, y_pos), 6, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor=color_loss, linewidth=3)
    ax.add_patch(loss_box)
    ax.text(5, y_pos + 0.5, 'Physics-Informed Loss', 
            fontsize=12, fontweight='bold', ha='center', color='darkred')
    ax.text(5, y_pos + 0.2, 'Loss = MSE(τ_predicted, τ_target)', 
            fontsize=10, ha='center', family='monospace')
    ax.text(5, y_pos - 0.1, '+ λ_GRF × regularization', 
            fontsize=9, ha='center', family='monospace')
    
    # Add target torques
    target_box = FancyBboxPatch((6.5, y_pos + 0.05), 2.5, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='white',
                               linewidth=2, linestyle='--')
    ax.add_patch(target_box)
    ax.text(7.75, y_pos + 0.35, 'τ_target', 
            fontsize=10, fontweight='bold', ha='center', color='red')
    ax.text(7.75, y_pos + 0.1, 'from tau.csv', 
            fontsize=8, ha='center', style='italic')
    
    # Arrow from target
    arrow_target = FancyArrowPatch((6.5, y_pos + 0.35), (5.2, y_pos + 0.4),
                                  arrowstyle='->', mutation_scale=25,
                                  linewidth=2, color='red', linestyle='--')
    ax.add_patch(arrow_target)
    
    # Gradient flow (backward pass)
    y_pos -= 1
    ax.text(5, y_pos, '═══════ GRADIENT FLOW (BACKPROPAGATION) ═══════', 
            fontsize=11, fontweight='bold', ha='center', color='green')
    
    # Big curved arrow showing backprop
    backprop_arrow = FancyArrowPatch((8.5, y_pos + 0.3), (8.5, 10.5),
                                    arrowstyle='<-', mutation_scale=40,
                                    linewidth=4, color='green', alpha=0.3,
                                    connectionstyle="arc3,rad=.3")
    ax.add_patch(backprop_arrow)
    ax.text(9.2, 6.5, 'Gradients\nflow back\nthrough all\nlayers', 
            fontsize=9, ha='center', color='green', fontweight='bold',
            rotation=-90, va='center')
    
    # Add legend
    legend_y = 0.5
    ax.text(5, legend_y, 'Key Insight: Gradients automatically flow through Jacobian multiplication!', 
            fontsize=10, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Add side notes
    ax.text(0.5, 7, 'Forward\nPass', fontsize=11, ha='center', 
            fontweight='bold', color='blue', rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig('Results/transformer_architecture.png', dpi=200, bbox_inches='tight')
    print("✓ Architecture diagram saved to Results/transformer_architecture.png")


def create_gradient_flow_diagram():
    """Create a diagram showing gradient computation."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Gradient Flow Through Physics Layer', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Forward pass
    y = 8
    ax.text(1, y, 'FORWARD PASS:', fontsize=12, fontweight='bold', color='blue')
    
    equations = [
        ('1.', 'Transformer predicts GRF/COP', 'F = Transformer(q, q̇, q̈)', 7.5),
        ('2.', 'Jacobian to torques', 'τ = J^T · F', 6.5),
        ('3.', 'Compute loss', 'L = ||τ - τ_target||²', 5.5),
    ]
    
    for num, desc, eq, y_pos in equations:
        ax.text(1.5, y_pos, num, fontsize=11, fontweight='bold')
        ax.text(2, y_pos, desc, fontsize=10)
        ax.text(6, y_pos, eq, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Backward pass
    y = 4
    ax.text(1, y, 'BACKWARD PASS (automatic via JAX):', fontsize=12, fontweight='bold', color='green')
    
    gradients = [
        ('1.', '∂L/∂τ = 2(τ - τ_target)', 3),
        ('2.', '∂τ/∂F = J^T', 2.3),
        ('3.', '∂L/∂F = ∂L/∂τ · ∂τ/∂F = 2(τ - τ_target) · J^T', 1.6),
        ('4.', '∂L/∂θ = ∂L/∂F · ∂F/∂θ  (transformer backprop)', 0.9),
    ]
    
    for num, eq, y_pos in gradients:
        ax.text(1.5, y_pos, num, fontsize=11, fontweight='bold')
        ax.text(2.2, y_pos, eq, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Key insight box
    insight_box = FancyBboxPatch((1, 0), 8, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='gold', facecolor='lightyellow',
                                linewidth=3)
    ax.add_patch(insight_box)
    ax.text(5, 0.3, '💡 Key: JAX computes all gradients automatically - you just write the forward pass!', 
            fontsize=11, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Results/gradient_flow_diagram.png', dpi=200, bbox_inches='tight')
    print("✓ Gradient flow diagram saved to Results/gradient_flow_diagram.png")


if __name__ == "__main__":
    print("Creating architecture diagrams...")
    create_architecture_diagram()
    create_gradient_flow_diagram()
    print("\n✓ All diagrams created!")
    print("\nGenerated files:")
    print("  - Results/transformer_architecture.png")
    print("  - Results/gradient_flow_diagram.png")
