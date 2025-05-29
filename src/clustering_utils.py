# clustering_utils.py

import os
import pickle

from src.config import (COLOR_LOG_FILE_PATH)
# Global in-memory log to collect entries across calls
color_transitions_log = []

def log_color_shifts(tokens, state_sequence, cmap=None):
    """
    Logs the positions where HMM state transitions (color changes) occur in a sequence.

    Args:
        tokens (List[str]): The tokenized input sentence.
        state_sequence (List[int]): The decoded HMM states per token.
        cmap (callable, optional): Optional colormap function for debugging or visualization.
    """
    if tokens is None or state_sequence is None or len(tokens) == 0 or len(state_sequence) == 0 or len(tokens) != len(state_sequence):
        return

    transitions = []
    states = []
    prev_state = state_sequence[0]
    start_idx = 0

    for i, state in enumerate(state_sequence):
        if state != prev_state:
            transitions.append((start_idx, i))
            states.append(prev_state)
            start_idx = i
            prev_state = state

    # Final segment
    transitions.append((start_idx, len(state_sequence)))
    states.append(state_sequence[-1])

    color_transitions_log.append({
        "tokens": tokens,
        "transitions": transitions,
        "states": states
    })

def save_color_log(path=COLOR_LOG_FILE_PATH):
    """
    Saves the accumulated color transitions to a .pkl file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(color_transitions_log, f)
    print("Color logs saved!")

def load_color_log(path=COLOR_LOG_FILE_PATH):
    """
    Loads color transitions from a previously saved file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

