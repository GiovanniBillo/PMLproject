
"""
Run script for logging inference trajectories on 10k IMDB reviews.

This script executes the inference logging process from the Inference_logging.ipynb notebook
as a standalone Python script that can be run from the terminal.

Usage: python run_inference_logging.py
"""

import sys
import os
import numpy as np
import torch
import argparse

# Add the src directory to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_NAME, DATASET_NAME, 
    NUM_TRAIN_SAMPLES, MAX_TOKENS, LOG_FILE_PATH, DEVICE
)
from data_utils import get_tokenizer, load_imdb_data, preprocess_data_for_inference_logging
from black_box_model import BlackBoxSentimentClassifier, log_inference_trajectories


def main():
    parser = argparse.ArgumentParser(description='Log inference trajectories for IMDB reviews')
    parser.add_argument('--num_samples', type=int, default=25000, 
                       help='Number of reviews to process (default: 25000)')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                       help=f'Maximum tokens per review (default: {MAX_TOKENS})')
    parser.add_argument('--output_path', type=str, default=LOG_FILE_PATH,
                       help=f'Output path for trajectories (default: {LOG_FILE_PATH})')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Dataset split to use (default: train)')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Processing {args.num_samples} reviews from {args.split} split")
    print(f"Max tokens per review: {args.max_tokens}")
    print(f"Output path: {args.output_path}")
    

    print("\n=== Initializing Black-Box Model ===")
    bb_model = BlackBoxSentimentClassifier(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = bb_model.tokenizer
    

    print(f"\n=== Loading {args.split} Data ===")
    imdb_data_raw = load_imdb_data(split=args.split, num_samples=args.num_samples, shuffle=True)
    print(f"Loaded {len(imdb_data_raw)} raw samples.")
    
    processed_data = preprocess_data_for_inference_logging(imdb_data_raw, tokenizer)
    print(f"Processed {len(processed_data)} samples for inference logging.")
    
   
    print("\n=== Logging Inference Trajectories ===")
    trajectories = log_inference_trajectories(processed_data, bb_model, max_len=args.max_tokens)
    
 
    trajectories = [t for t in trajectories if t.shape[0] > 0]
    
    print(f"Generated {len(trajectories)} trajectories.")
    if trajectories:
        print(f"Example trajectory shape: {trajectories[0].shape}")
        print(f"Example trajectory first 3 steps:\n{trajectories[0][:3]}")
    
   
    print(f"\n=== Saving Trajectories ===")
    
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if trajectories:
      
        np.savez_compressed(args.output_path, *trajectories)
        print(f"Saved {len(trajectories)} trajectories to {args.output_path}")
        
  
        print("\n=== Verifying Save/Load ===")
        loaded_data = np.load(args.output_path, allow_pickle=True)
        loaded_trajectories = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]
        print(f"Loaded back {len(loaded_trajectories)} trajectories.")
        
        assert len(loaded_trajectories) == len(trajectories), "Mismatch in number of trajectories"
        if loaded_trajectories:
            assert np.allclose(loaded_trajectories[0], trajectories[0]), "First trajectory mismatch"
            print("Verification successful!")
    else:
        print("No trajectories to save.")
    
    print("\n=== Process Complete ===")


if __name__ == "__main__":
    main()