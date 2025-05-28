import numpy as np
from hmmlearn import hmm
import joblib
from tqdm.auto import tqdm

from .config import NUM_HMM_STATES, HMM_N_ITER, HMM_TOL, HMM_COV_TYPE, \
                       TARGET_SENTIMENT, HMM_MODEL_PATH, PROB_THRESHOLDS

class HMMSurrogate:
    def __init__(self, n_states=NUM_HMM_STATES, n_iter=HMM_N_ITER, tol=HMM_TOL,
                 covariance_type=HMM_COV_TYPE, random_state=2):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=True # To see Baum-Welch progress
        )
        self.is_trained = False

    def train(self, observation_sequences):
        """
        Trains the HMM on a list of observation sequences.
        Args:
            observation_sequences: A list of 2D numpy arrays.
                                   Each array has shape (T_r, D_obs),
                                   where T_r is sequence length for review r,
                                   and D_obs is observation dimension (num_classes).
        """
        # hmmlearn expects a single concatenated array and an array of lengths
        if not observation_sequences:
            print("Error: No observation sequences provided for training.")
            return

        lengths = [seq.shape[0] for seq in observation_sequences if seq.shape[0] > 0]
        if not lengths:
            print("Error: All observation sequences are empty.")
            return
            
        concatenated_sequences = np.concatenate([seq for seq in observation_sequences if seq.shape[0] > 0])
        
        print(f"Training HMM with {self.n_states} states on {len(lengths)} sequences.")
        print(f"Total observations: {concatenated_sequences.shape[0]}, Observation dim: {concatenated_sequences.shape[1]}")
        
        self.model.fit(concatenated_sequences, lengths)
        self.is_trained = True
        print("HMM training complete.")
        self.print_hmm_parameters()

    def print_hmm_parameters(self):
        if not self.is_trained:
            print("HMM is not trained yet.")
            return
        print("\n--- HMM Parameters ---")
        print("Initial state probabilities (startprob_):\n", self.model.startprob_)
        print("\nTransition matrix (transmat_):\n", self.model.transmat_)
        print("\nEmission probabilities (means_ for GaussianHMM):\n", self.model.means_)
        print("\nEmission covariances (covars_ for GaussianHMM):\n", self.model.covars_)


    def decode_sequence(self, observation_sequence):
        """
        Decodes the most likely hidden state sequence for a given observation sequence.
        Args:
            observation_sequence: A 2D numpy array (T, D_obs).
        Returns:
            A 1D numpy array of hidden states.
        """
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")
        if observation_sequence.ndim == 1: # If a single observation
            observation_sequence = observation_sequence.reshape(1, -1)
        if observation_sequence.shape[0] == 0:
            return np.array([], dtype=int)
            
        _log_prob, state_sequence = self.model.decode(observation_sequence, algorithm="viterbi")
        return state_sequence

    def save_model(self, path=HMM_MODEL_PATH):
        if not self.is_trained:
            print("Warning: Saving an untrained HMM model.")
        joblib.dump(self.model, path)
        print(f"HMM model saved to {path}")

    def load_model(self, path=HMM_MODEL_PATH):
        self.model = joblib.load(path)
        self.is_trained = True # Assume loaded model is trained
        self.n_states = self.model.n_components
        print(f"HMM model loaded from {path}")
        self.print_hmm_parameters() # Print params of loaded model

    def analyze_states(self, observation_sequences, decoded_state_sequences, target_class_idx=TARGET_SENTIMENT):
        """
        Analyzes HMM states based on observations and decoded states.
        Args:
            observation_sequences: List of original observation sequences (probs).
            decoded_state_sequences: List of corresponding decoded HMM state sequences.
            target_class_idx: Index of the class probability to focus on (e.g., P(positive)).
                               Defaults to TARGET_SENTIMENT from config.
        Returns:
            A dictionary where keys are state indices and values are dicts
            containing 'avg_prob_target_class', 'num_occurrences'.
            The dictionary also includes a 'state_names' key mapping HMM state indices
            to their interpreted names.
        """
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")

        # Assumes PROB_THRESHOLDS is imported, e.g., from .config import PROB_THRESHOLDS
        # And TARGET_SENTIMENT is also available (as it's a default arg).

        state_analysis = {i: {'occurrences': 0, 'sum_target_prob': 0.0} for i in range(self.n_states)}

        for obs_seq, state_seq in zip(observation_sequences, decoded_state_sequences):
            if obs_seq.shape[0] == 0 or state_seq.shape[0] == 0: continue
            for i in range(len(state_seq)):
                state = state_seq[i]
                prob_vector = obs_seq[i]
                state_analysis[state]['occurrences'] += 1
                state_analysis[state]['sum_target_prob'] += prob_vector[target_class_idx]
        
        results = {}
        print("\n--- HMM State Analysis ---")
        for state, data in state_analysis.items():
            avg_prob = (data['sum_target_prob'] / data['occurrences']) if data['occurrences'] > 0 else 0.0
            results[state] = {
                'avg_prob_target_class': avg_prob,
                'num_occurrences': data['occurrences']
            }
            print(f"State {state}: Occurrences = {data['occurrences']}, Avg. P(Class {target_class_idx}) = {avg_prob:.3f}")
        
        state_names = {}
        # Use PROB_THRESHOLDS for categorization.
        # Example: PROB_THRESHOLDS = {"NEG_OBS": 0.4, "NEU_OBS": 0.6}
        for state_idx, data_dict in results.items():
            avg_prob = data_dict['avg_prob_target_class']
            
            if avg_prob <= PROB_THRESHOLDS["NEG_OBS"]:
                state_names[state_idx] = "Leaning Negative/Low Confidence"
            elif avg_prob <= PROB_THRESHOLDS["NEU_OBS"]: # i.e., NEG_OBS < avg_prob <= NEU_OBS
                state_names[state_idx] = "Neutral/Uncertain"
            else: # avg_prob > PROB_THRESHOLDS["NEU_OBS"]
                state_names[state_idx] = "Leaning Positive/High Confidence"
        
        print("\nSuggested State Interpretations (based on P(Class {target_class_idx})):")
        
        print_info = []
        for state_idx, data_dict in results.items():
            # Ensure we are processing actual HMM state data (integer keys)
            if isinstance(state_idx, int):
                avg_prob_val = data_dict['avg_prob_target_class']
                name = state_names[state_idx] # Get the name from the map created above
                print_info.append((state_idx, avg_prob_val, name))
            
        # Sort by average probability of the target class for clearer presentation
        print_info.sort(key=lambda x: x[1])

        for state_idx_sorted, avg_prob_sorted, name_sorted in print_info:
            print(f"  HMM State {state_idx_sorted}: ~{name_sorted} (Avg. P(Class {target_class_idx}) = {avg_prob_sorted:.3f})")
        
        results['state_names'] = state_names # Add the interpretation map to the results
        return results

if __name__ == '__main__':
    # Example: Simulate some observation sequences
    # Each sequence: (num_steps, num_features=2 for binary classification probs)
    # For example, prob_class_0, prob_class_1
    obs1 = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    obs2 = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2], [0.9, 0.1], [0.95, 0.05]])
    obs3 = np.array([[0.5,0.5], [0.4,0.6], [0.6,0.4]])
    all_obs = [obs1, obs2, obs3]

    hmm_surrogate = HMMSurrogate(n_states=3)
    hmm_surrogate.train(all_obs)
    
    # Decode a sequence
    test_obs = np.array([[0.8, 0.2], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
    decoded_states = hmm_surrogate.decode_sequence(test_obs)
    print(f"\nTest Observation Sequence:\n{test_obs}")
    print(f"Decoded State Sequence: {decoded_states}")

    # Analyze states
    decoded_all = [hmm_surrogate.decode_sequence(s) for s in all_obs]
    state_analysis_results = hmm_surrogate.analyze_states(all_obs, decoded_all, target_class_idx=1)

    hmm_surrogate.save_model("models/dummy_hmm.pkl")
    loaded_hmm = HMMSurrogate()
    loaded_hmm.load_model("models/dummy_hmm.pkl")
    assert np.allclose(loaded_hmm.model.transmat_, hmm_surrogate.model.transmat_)