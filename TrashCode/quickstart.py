#!/usr/bin/env python3
"""
Quick Start Script for Physics-Informed Transformer Training

This script helps you get started quickly by:
1. Checking if required data files exist
2. Installing dependencies if needed
3. Running a test forward pass
4. Optionally starting training

Usage:
    python quickstart.py
"""

import sys
from pathlib import Path
import subprocess


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required = {
        'jax': 'jax[cuda]',  # or jax[cpu] for CPU only
        'flax': 'flax',
        'optax': 'optax',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for package, install_name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(install_name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")
        if response.lower() == 'y':
            print("\nInstalling packages...")
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
            print("✓ Installation complete!")
        else:
            print("\nPlease install manually:")
            print(f"  pip install {' '.join(missing)}")
            return False
    
    return True


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")
    
    required_files = [
        "Results/qpos_matrix.npy",
        "Results/qvel_matrix.npy",
        "Results/qacc_matrix.npy",
        "Results/jacobian_data.npz",
        "Results/grf_matrix.npy",
        "Results/cop_matrix.npy"
    ]
    
    optional_files = [
        "Results/jax_joint_forces.npy",
        "PatientData/Falisse_2017_subject_01/tau.csv"
    ]
    
    all_good = True
    for filepath in required_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (MISSING - REQUIRED)")
            all_good = False
    
    print("\nOptional files (at least one needed for target torques):")
    has_optional = False
    for filepath in optional_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
            has_optional = True
        else:
            print(f"  ✗ {filepath}")
    
    if not all_good:
        print("\n❌ Missing required data files!")
        print("\nTo generate these files, you need to:")
        print("  1. Run your main inverse dynamics script (MJX_RunID.py)")
        print("  2. Ensure it saves all required matrices to Results/")
        return False
    
    if not has_optional:
        print("\n⚠ No target torque files found!")
        print("You need either:")
        print("  - Results/jax_joint_forces.npy (from inverse dynamics)")
        print("  - PatientData/Falisse_2017_subject_01/tau.csv (reference data)")
        return False
    
    return True


def test_model():
    """Test that the model can be initialized and run."""
    print("\nTesting model initialization...")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        from TrashCode.physics_informed_transformer import (
            GRFCOPTransformer,
            create_train_state,
            train_step
        )
        
        # Model parameters
        batch_size = 4
        seq_len = 50
        nv = 37
        input_dim = 3 * nv
        
        # Initialize model
        model = GRFCOPTransformer(
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            output_dim=12,
            dropout_rate=0.1
        )
        
        # Create training state
        rng = random.PRNGKey(0)
        state = create_train_state(
            rng,
            model,
            input_shape=(batch_size, seq_len, input_dim),
            learning_rate=1e-4
        )
        
        print("  ✓ Model initialized successfully")
        
        # Test forward pass
        dummy_input = jnp.ones((batch_size, seq_len, input_dim))
        predictions = model.apply(
            {'params': state.params},
            dummy_input,
            train=False
        )
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {dummy_input.shape}")
        print(f"    Output shape: {predictions.shape}")
        
        # Test training step
        dummy_batch = {
            'kinematics': dummy_input,
            'target_torques': jnp.zeros((batch_size, seq_len, nv))
        }
        dummy_jacobian = jnp.ones((seq_len, nv, 12))
        body_ids = {'calcn_l': 10, 'calcn_r': 15, 'pelvis': 1}
        
        new_state, metrics = train_step(
            state,
            dummy_batch,
            dummy_jacobian,
            body_ids
        )
        
        print("  ✓ Training step successful")
        print(f"    Loss: {metrics['total_loss']:.4f}")
        print(f"    Gradient norm: {metrics['grad_norm']:.4f}")
        
        # Check GPU
        devices = jax.devices()
        print(f"\n  Available devices: {devices}")
        if any('gpu' in str(d).lower() for d in devices):
            print("  ✓ GPU detected - training will be fast!")
        else:
            print("  ⚠ No GPU detected - training will be slower")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_menu():
    """Show main menu."""
    print("\n" + "="*70)
    print("Physics-Informed Transformer - Quick Start")
    print("="*70)
    print("\nWhat would you like to do?")
    print("  1. Validate inverse dynamics (compare with tau.csv)")
    print("  2. Start training with default parameters")
    print("  3. Start training with custom parameters")
    print("  4. Visualize existing training results")
    print("  5. Exit")
    print()
    
    choice = input("Enter choice (1-5): ")
    return choice


def run_validation():
    """Run inverse dynamics validation."""
    print("\nRunning validation...")
    subprocess.run([sys.executable, "validate_jax_inverse_dynamics.py"])


def run_training(custom=False):
    """Run training."""
    if custom:
        print("\nCustom training parameters:")
        epochs = input("  Number of epochs [100]: ") or "100"
        batch_size = input("  Batch size [16]: ") or "16"
        lr = input("  Learning rate [1e-4]: ") or "1e-4"
        
        cmd = [
            sys.executable,
            "train_physics_transformer.py",
            "--epochs", epochs,
            "--batch_size", batch_size,
            "--lr", lr
        ]
    else:
        print("\nStarting training with default parameters...")
        print("  Epochs: 100")
        print("  Batch size: 16")
        print("  Learning rate: 1e-4")
        print("  Sequence length: 200")
        
        cmd = [
            sys.executable,
            "train_physics_transformer.py",
            "--epochs", "100",
            "--batch_size", "16"
        ]
    
    subprocess.run(cmd)


def run_visualization():
    """Run visualization."""
    print("\nVisualizing training results...")
    subprocess.run([sys.executable, "visualize_training.py"])


def main():
    """Main function."""
    print("="*70)
    print("Physics-Informed Transformer for GRF/COP Prediction")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first")
        return
    
    # Check data files
    if not check_data_files():
        print("\n❌ Please generate required data files first")
        print("\nSee TRANSFORMER_TRAINING_README.md for details")
        return
    
    # Test model
    if not test_model():
        print("\n❌ Model test failed")
        return
    
    print("\n" + "="*70)
    print("✓ All checks passed! Ready to train.")
    print("="*70)
    
    # Main menu loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_validation()
        elif choice == '2':
            run_training(custom=False)
        elif choice == '3':
            run_training(custom=True)
        elif choice == '4':
            run_visualization()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("\n⚠ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
