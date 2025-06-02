#!/usr/bin/env python3
"""
HMM Surrogate Model Comparison Script

This script compares the performance of a main HMM trained on full data
vs a dummy HMM trained on a subset of data using fidelity metrics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from config import (
    MODEL_NAME, NUM_TEST_SAMPLES, MAX_TOKENS, TARGET_SENTIMENT, DEVICE
)
from hmm_surrogate import HMMSurrogate
from black_box_model import BlackBoxSentimentClassifier, log_inference_trajectories
from data_utils import load_imdb_data, preprocess_data_for_inference_logging
from visualization_utils import plot_hmm_transition_matrix, plot_avg_probabilities_per_state

def load_training_trajectories():
    """Load training trajectories from saved file or generate them."""
    log_file_path = "/home/gamerio/Documents/pml/PMLproject/notebooks/data/imbd_inference_logs25k.npz"
    
    try:
        print(f"Loading training trajectories from {log_file_path}...")
        loaded_data = np.load(log_file_path, allow_pickle=True)
        train_trajectories = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]
        print(f"Loaded {len(train_trajectories)} training trajectories.")
        return train_trajectories
    except FileNotFoundError:
        print(f"Training trajectories file not found at {log_file_path}")
        print("Please run the inference logging notebook first to generate training data.")
        return []

def load_main_hmm_model():
    """Load the main HMM model if it exists."""
    model_path = "/home/gamerio/Documents/pml/PMLproject/notebooks/models/sentiment_hmm25k_4states.pkl"
    
    try:
        print(f"Loading main HMM model from {model_path}...")
        hmm_surrogate_model = HMMSurrogate().load_model(model_path)
        print("Main HMM model loaded successfully.")
        return hmm_surrogate_model
    except FileNotFoundError:
        print(f"Main HMM model not found at {model_path}")
        print("Please train the main HMM model first.")
        return None

def train_dummy_hmm(train_trajectories, num_samples=1000, n_states=4):
    """Train a dummy HMM on a subset of training data."""
    print("\n--- Training Dummy HMM ---")
    
    if not train_trajectories or len(train_trajectories) < num_samples:
        print(f"Not enough training trajectories ({len(train_trajectories)}) to train dummy HMM with {num_samples} samples.")
        return None, []
    
    print(f"Training Dummy HMM with {n_states} states on {num_samples} trajectories...")
    dummy_train_set = train_trajectories[:num_samples]
    
    try:
        dummy_hmm_model = HMMSurrogate(
            n_states=n_states,
            n_iter=50,
            random_state=2,
            covariance_type='full'
        )
        dummy_hmm_model.train(dummy_train_set)
        print("Dummy HMM training complete.")
        return dummy_hmm_model, dummy_train_set
        
    except ValueError as e:
        print(f"Error training Dummy HMM: {e}")
        print("This can happen if covariance matrices are not positive-definite, often due to insufficient data.")
        print("Consider using 'diag' or 'spherical' covariance_type, or increasing the number of training samples.")
        return None, []
        
    except Exception as e:
        print(f"An unexpected error occurred during Dummy HMM training: {e}")
        return None, []

def analyze_dummy_hmm(dummy_hmm_model, dummy_train_set):
    """Analyze and visualize dummy HMM parameters."""
    if not dummy_hmm_model or not dummy_hmm_model.is_trained or not dummy_train_set:
        print("Skipping Dummy HMM analysis - model not available or training set empty.")
        return None
    
    print("\n--- Analyzing Dummy HMM Parameters ---")
    
    # Decode states for the dummy training set
    dummy_decoded_train_states = [
        dummy_hmm_model.decode_sequence(traj) 
        for traj in dummy_train_set if traj.shape[0] > 0
    ]
    
    # Filter out empty sequences
    valid_dummy_train_trajectories = [
        traj for traj, states in zip(dummy_train_set, dummy_decoded_train_states) 
        if states.size > 0
    ]
    valid_dummy_decoded_train_states = [
        states for states in dummy_decoded_train_states if states.size > 0
    ]
    
    if not valid_dummy_train_trajectories or not valid_dummy_decoded_train_states:
        print("No valid trajectories/states for Dummy HMM analysis after decoding.")
        return None
    
    dummy_state_analysis_results = dummy_hmm_model.analyze_states(
        valid_dummy_train_trajectories,
        valid_dummy_decoded_train_states,
        target_class_idx=TARGET_SENTIMENT
    )
    
    # Create visualizations
    print("\nGenerating Dummy HMM visualizations...")
    
    # Transition matrix
    plt.figure(figsize=(8, 6))
    plot_hmm_transition_matrix(
        dummy_hmm_model.model, 
        state_names=dummy_state_analysis_results.get('state_names')
    )
    plt.savefig('dummy_hmm_transition_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Average probabilities per state
    plt.figure(figsize=(8, 5))
    plot_avg_probabilities_per_state(
        dummy_state_analysis_results, 
        target_class_idx=TARGET_SENTIMENT
    )
    plt.savefig('dummy_hmm_avg_probabilities.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Dummy HMM visualizations saved as PNG files.")
    return dummy_state_analysis_results

def prepare_fidelity_test_set(num_test_samples=20):
    """Prepare a dedicated test set for fidelity evaluation."""
    print(f"\n--- Preparing Fidelity Test Set ---")
    print(f"Preparing test set with {num_test_samples} samples from IMDB 'test' split...")
    
    try:
        bb_model_for_fidelity_test = BlackBoxSentimentClassifier(model_name=MODEL_NAME, device=DEVICE)
        tokenizer_for_fidelity_test = bb_model_for_fidelity_test.tokenizer
        
        imdb_fidelity_test_raw = load_imdb_data(split='test', num_samples=num_test_samples, shuffle=True)
        processed_fidelity_test_data = preprocess_data_for_inference_logging(
            imdb_fidelity_test_raw, tokenizer_for_fidelity_test
        )
        fidelity_evaluation_set = log_inference_trajectories(
            processed_fidelity_test_data, bb_model_for_fidelity_test, max_len=MAX_TOKENS
        )
        
        # Filter out empty trajectories
        fidelity_evaluation_set = [t for t in fidelity_evaluation_set if t.shape[0] > 0]
        
        print(f"Generated {len(fidelity_evaluation_set)} test trajectories for fidelity evaluation.")
        return fidelity_evaluation_set
        
    except Exception as e:
        print(f"Error preparing fidelity test set: {e}")
        return []

def calculate_fidelity_metrics(model, model_name, fidelity_evaluation_set):
    """Calculate fidelity metrics for a given model."""
    if not model or not model.is_trained or not fidelity_evaluation_set:
        print(f"{model_name} not ready or no evaluation set for fidelity.")
        return None
    
    print(f"\nCalculating Fidelity for {model_name}...")
    try:
        fidelity_results = model.calculate_fidelity_metrics(fidelity_evaluation_set)
        return fidelity_results
    except Exception as e:
        print(f"Error calculating fidelity for {model_name}: {e}")
        return None

def print_fidelity_comparison(main_results, dummy_results, num_dummy_samples):
    """Print comprehensive fidelity comparison results."""
    print("\n\n--- Fidelity Comparison Summary ---")
    
    if main_results:
        print(f"\nMain HMM (trained on full training data):")
        print(f"  Avg KL Divergence: {main_results.get('avg_kl_divergence', float('nan')):.4f}")
        print(f"  Avg NLL:           {main_results.get('avg_nll', float('nan')):.4f}")
        print(f"  Steps Evaluated:   {main_results.get('total_steps_evaluated', 0)}")
    else:
        print("\nMain HMM: Fidelity results not available.")
    
    if dummy_results:
        print(f"\nDummy HMM (trained on {num_dummy_samples} training samples):")
        print(f"  Avg KL Divergence: {dummy_results.get('avg_kl_divergence', float('nan')):.4f}")
        print(f"  Avg NLL:           {dummy_results.get('avg_nll', float('nan')):.4f}")
        print(f"  Steps Evaluated:   {dummy_results.get('total_steps_evaluated', 0)}")
    else:
        print("\nDummy HMM: Fidelity results not available.")
    
    if main_results and dummy_results:
        print("\nComparison Insights:")
        
        # Get values and handle NaN
        main_kl = main_results.get('avg_kl_divergence', float('inf'))
        dummy_kl = dummy_results.get('avg_kl_divergence', float('inf'))
        main_nll = main_results.get('avg_nll', float('inf'))
        dummy_nll = dummy_results.get('avg_nll', float('inf'))
        
        # Handle NaN values
        main_kl = float('inf') if np.isnan(main_kl) else main_kl
        dummy_kl = float('inf') if np.isnan(dummy_kl) else dummy_kl
        main_nll = float('inf') if np.isnan(main_nll) else main_nll
        dummy_nll = float('inf') if np.isnan(dummy_nll) else dummy_nll
        
        # KL Divergence comparison
        if main_kl < dummy_kl:
            print("  - Main HMM has LOWER (better) KL Divergence than Dummy HMM.")
        elif main_kl == dummy_kl and main_kl != float('inf'):
            print("  - Main HMM has EQUAL KL Divergence as Dummy HMM.")
        elif main_kl == float('inf') and dummy_kl == float('inf'):
            print("  - KL Divergence for both Main and Dummy HMM is NaN/Inf.")
        else:
            print("  - Main HMM has HIGHER (worse) KL Divergence compared to Dummy HMM, or one/both is NaN/Inf.")
        
        # NLL comparison
        if main_nll < dummy_nll:
            print("  - Main HMM has LOWER (better) NLL than Dummy HMM.")
        elif main_nll == dummy_nll and main_nll != float('inf'):
            print("  - Main HMM has EQUAL NLL as Dummy HMM.")
        elif main_nll == float('inf') and dummy_nll == float('inf'):
            print("  - NLL for both Main and Dummy HMM is NaN/Inf.")
        else:
            print("  - Main HMM has HIGHER (worse) NLL compared to Dummy HMM, or one/both is NaN/Inf.")
        
        print("  (Lower KL and NLL values indicate a more faithful surrogate model.)")
        
        # Additional analysis
        if dummy_kl < main_kl and dummy_nll < main_nll:
            print("\n  *** POTENTIAL ISSUE DETECTED ***")
            print(f"  The dummy HMM (trained on only {num_dummy_samples} samples) appears to perform")
            print("  better than the main HMM on both metrics. This suggests:")
            print("  1. Possible data leakage or evaluation issues")
            print("  2. Overfitting in the main model")
            print("  3. Insufficient test data for reliable evaluation")
            print("  4. Numerical instabilities in the training process")

def main():
    """Main execution function."""
    print("=== HMM Surrogate Model Comparison ===")
    
    # Configuration
    NUM_DUMMY_SAMPLES = 1000 
    DUMMY_N_STATES = 4
    
    # Load training trajectories
    train_trajectories = load_training_trajectories()
    if not train_trajectories:
        return
    
    # Load main HMM model
    hmm_surrogate_model = load_main_hmm_model()
    
    # Train dummy HMM
    dummy_hmm_model, dummy_train_set = train_dummy_hmm(
        train_trajectories, 
        num_samples=NUM_DUMMY_SAMPLES,
        n_states=DUMMY_N_STATES
    )
    
    # Analyze dummy HMM if training was successful
    if dummy_hmm_model:
        analyze_dummy_hmm(dummy_hmm_model, dummy_train_set)
    
    # Prepare fidelity test set
    fidelity_evaluation_set = prepare_fidelity_test_set(NUM_TEST_SAMPLES)
    
    # Calculate fidelity metrics for both models
    main_hmm_fidelity_results = calculate_fidelity_metrics(
        hmm_surrogate_model, "Main HMM", fidelity_evaluation_set
    )
    
    dummy_hmm_fidelity_results = calculate_fidelity_metrics(
        dummy_hmm_model, "Dummy HMM", fidelity_evaluation_set
    )
    
    # Print comparison results
    print_fidelity_comparison(
        main_hmm_fidelity_results, 
        dummy_hmm_fidelity_results, 
        NUM_DUMMY_SAMPLES
    )
    
    print("\n=== Script completed ===")

if __name__ == "__main__":
    main()