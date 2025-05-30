# convergence_utils.py

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

def plot_log_likelihoods(log_likelihoods, n_iter=None, show=True, save_path=None):
    """
    Plots the log-likelihood progression over iterations.

    Args:
        log_likelihoods: List of log-likelihood values.
        n_iter: Optional number of iterations (defaults to len(log_likelihoods)).
        show: If True, displays the plot interactively.
        save_path: If provided, saves the plot as a PNG file.
    """
    if not log_likelihoods:
        print("No log-likelihoods to plot.")
        return
    
    if n_iter is None:
        n_iter = len(log_likelihoods)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_iter + 1), log_likelihoods, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('HMM Training Convergence')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Convergence plot saved to '{save_path}'.")
    elif show:
        plt.show()
    else:
        plt.close()

def save_convergence_data(convergence_data, diagnostics_dir="convergence_diagnostics"):
    """
    Saves convergence diagnostics data as a pickle file.

    Args:
        convergence_data: Dictionary containing convergence info.
        diagnostics_dir: Directory where the file will be saved.
    """
    os.makedirs(diagnostics_dir, exist_ok=True)
    data_filename = os.path.join(diagnostics_dir, "convergence_data.pkl")
    with open(data_filename, 'wb') as f:
        pickle.dump(convergence_data, f)
    print(f"Convergence data saved to '{data_filename}'.")

def plot_state_chain_acf(state_sequences, max_lag=100, show_plot=True, save_path=None):
    """
    Plots the autocorrelation function of the hidden state chains.

    Args:
        state_sequences: List of 1D numpy arrays representing decoded hidden state sequences.
        max_lag: Maximum lag to compute ACF for.
        show_plot: If True, displays the plot.
        save_path: If provided, saves the plot to this path.
    """
    if not state_sequences:
        print("No state sequences provided.")
        return

    # Concatenate all state sequences
    concatenated_states = np.concatenate(state_sequences)

    n = len(concatenated_states)
    mean_state = np.mean(concatenated_states)
    var_state = np.var(concatenated_states)
    acf = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        cov = np.sum((concatenated_states[:n - lag] - mean_state) *
                     (concatenated_states[lag:] - mean_state)) / (n - lag)
        acf[lag] = cov / var_state

    # Plot
    plt.figure(figsize=(8, 5))
    plt.stem(range(max_lag + 1), acf)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function of Hidden State Chains")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"ACF plot saved to '{save_path}'.")
    elif show_plot:
        plt.show()
    else:
        plt.close()

def mean_run_length(state_sequence):
    """
    Computes the mean run length of consecutive states.
    """
    if len(state_sequence) == 0:
        return 0.0  # Return 0 instead of nan

    runs = []
    current_run = 1
    for i in range(1, len(state_sequence)):
        if state_sequence[i] == state_sequence[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return np.mean(runs) if runs else 0.0


def posterior_predictive_check(state_sequences, summary_func, observed_value, num_simulations=1000):
    """
    Performs a posterior predictive check on the hidden state sequences.

    Args:
        state_sequences: List of 1D numpy arrays (decoded state sequences).
        summary_func: Function to compute a summary statistic from a state sequence.
        observed_value: The summary statistic computed on your observed data.
        num_simulations: Number of simulations to compare.

    Returns:
        p_value: The proportion of simulations with summary >= observed_value.
    """
    simulated_summaries = []
    non_empty_sequences = [seq for seq in state_sequences if len(seq) > 0]

    if not non_empty_sequences:
        print("Error: All state sequences are empty. Cannot run PPC.")
        return np.nan, []

    for seq in non_empty_sequences:
        repeats = max(1, num_simulations // len(non_empty_sequences))
        for _ in range(repeats):
            permuted_seq = np.random.permutation(seq)
            stat = summary_func(permuted_seq)
            simulated_summaries.append(stat)

    if not simulated_summaries:
        print("Warning: No simulated summaries were computed.")
        return np.nan, []

    simulated_summaries = np.array(simulated_summaries)
    p_value = np.mean(simulated_summaries >= observed_value)
    print(f"Observed summary: {observed_value:.3f}")
    print(f"Mean simulated summary: {np.mean(simulated_summaries):.3f}")
    print(f"Posterior predictive p-value: {p_value:.3f}")

    return p_value, simulated_summaries

def simulate_state_sequences(hmm_model, sequence_lengths, num_simulations=1000):
    """
    Generates simulated hidden state sequences from a trained HMM.

    Args:
        hmm_model: Trained hmmlearn model.
        sequence_lengths: List of sequence lengths to simulate.
        num_simulations: Total number of simulated sequences (approximately).

    Returns:
        simulated_sequences: List of 1D numpy arrays of simulated state sequences.
    """
    simulated_sequences = []
    for length in sequence_lengths:
        if length > 1:
            _, states  = hmm_model.sample(length)
            simulated_sequences.append(states.flatten())
    return simulated_sequences

def posterior_predictive_check_model_based(
    observed_state_sequences,
    simulated_state_sequences,
    summary_func,
    observed_value
):
    """
    Performs a posterior predictive check using model-based simulations.

    Args:
        observed_state_sequences: List of 1D numpy arrays (decoded observed sequences).
        simulated_state_sequences: List of 1D numpy arrays (generated by model.sample()).
        summary_func: Function to compute a summary statistic from a state sequence.
        observed_value: The summary statistic computed on your observed data.

    Returns:
        p_value: The proportion of simulated summaries >= observed_value.
    """
    simulated_summaries = [
        summary_func(seq) for seq in simulated_state_sequences if len(seq) > 0
    ]
    if not simulated_summaries:
        print("Warning: No simulated summaries were computed.")
        return np.nan, []

    simulated_summaries = np.array(simulated_summaries)
    p_value = np.mean(simulated_summaries >= observed_value)
    print(f"Observed summary: {observed_value:.3f}")
    print(f"Mean simulated summary: {np.mean(simulated_summaries):.3f}")
    print(f"Posterior predictive p-value: {p_value:.3f}")

    return p_value, simulated_summaries

