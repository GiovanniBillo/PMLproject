import numpy as np
from hmmlearn import hmm
import joblib
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal # For Gaussian PDF
from scipy.special import rel_entr # For KL divergence element-wise

from src.config import NUM_HMM_STATES, HMM_N_ITER, HMM_TOL, HMM_COV_TYPE, \
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
            transmat_prior=0.10,
        
            random_state=random_state,
            verbose=True, # Set to False for cleaner output during fidelity calculation
            min_covar=1e-2 # min_covar was added, good for stability
        )
        self.is_trained = False

    def train(self, observation_sequences):
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

        # Temporarily disable verbose during fit if it's too noisy for bulk training
        original_verbose = self.model.verbose
        self.model.verbose = False
        self.model.fit(concatenated_sequences, lengths)
        self.model.verbose = original_verbose # Restore verbose setting

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
        # Only print covars if they exist (hmmlearn might not populate if fit fails or is interrupted)
        if hasattr(self.model, 'covars_'):
            print("\nEmission covariances (covars_ for GaussianHMM):\n", self.model.covars_)
        else:
            print("\nEmission covariances (covars_ for GaussianHMM): Not available")


    def decode_sequence(self, observation_sequence):
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")
        if observation_sequence.ndim == 1:
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
        self.is_trained = True
        self.n_states = self.model.n_components
        print(f"HMM model loaded from {path}")
        # self.print_hmm_parameters() # Can be verbose, optionally call it after loading
        return self # Return self to allow chaining like model = HMMSurrogate().load_model()

    def analyze_states(self, observation_sequences, decoded_state_sequences, target_class_idx=TARGET_SENTIMENT):
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")

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
        for state_idx, data_dict in results.items():
            avg_prob = data_dict['avg_prob_target_class']
            if avg_prob <= PROB_THRESHOLDS["NEG_OBS"]: # Using PROB_THRESHOLDS from config
                state_names[state_idx] = "Leaning Negative/Low Confidence"
            elif avg_prob <= PROB_THRESHOLDS["NEU_OBS"]:
                state_names[state_idx] = "Neutral/Uncertain"
            else:
                state_names[state_idx] = "Leaning Positive/High Confidence"

        print("\nSuggested State Interpretations (based on P(Class {target_class_idx})):")
        print_info = []
        for state_idx, data_dict in results.items():
            if isinstance(state_idx, int):
                avg_prob_val = data_dict['avg_prob_target_class']
                name = state_names.get(state_idx, f"State {state_idx}") # Use .get for safety
                print_info.append((state_idx, avg_prob_val, name))

        print_info.sort(key=lambda x: x[1])
        for state_idx_sorted, avg_prob_sorted, name_sorted in print_info:
            print(f"  HMM State {state_idx_sorted}: ~{name_sorted} (Avg. P(Class {target_class_idx}) = {avg_prob_sorted:.3f})")

        results['state_names'] = state_names
        return results

    def predict_next_observation_distribution(self, obs_prefix):
        """
        Predicts the probability distribution of the next observation
        given the prefix of observations.
        Args:
            obs_prefix: A 2D numpy array (T_prefix, D_obs) of past observations.
        Returns:
            A 1D numpy array (D_obs,) representing the predicted probability
            distribution for the next observation. Returns None if prediction is not possible.
        """
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")
        

        # 1. Run forward algorithm on obs_prefix to get  log_alpha_t
        # hmmlearn's _forward_pass returns logalpha.
        # We need P(S_t=i | o_1...o_t), which is related to alpha but often obtained
        # from _compute_posteriors which gives P(S_t=i, o_1...o_t).
        # A simpler way: get P(S_t | O_1..O_t) from `predict_proba`
        # `predict_proba` gives P(S_t=i | o_1...o_t) for the *last* t in obs_prefix
        
        if obs_prefix.ndim == 1:
            obs_prefix = obs_prefix.reshape(1,-1)

        # state_posteriors_at_t will be P(S_t = i | o_1...o_t) for the last t
        try:
            state_posteriors_at_t = self.model.predict_proba(obs_prefix)[-1, :] # Shape: (n_states,)
        except Exception as e:
            # This can happen if obs_prefix is too short or causes issues in hmmlearn
            # print(f"Could not compute predict_proba for prefix of shape {obs_prefix.shape}: {e}")
            return None


        # 2. Calculate P(S_{t+1}=j | o_1...o_t) = sum_i P(S_t=i | o_1...o_t) * A_ij
        # P(S_t=i | o_1...o_t) is state_posteriors_at_t[i]
        # A_ij is self.model.transmat_[i, j]
        prob_s_next = np.zeros(self.n_states) # P(S_{t+1}=j | o_1...o_t)
        for j in range(self.n_states): # For each next state j
            sum_val = 0
            for i in range(self.n_states): # Sum over current states i
                sum_val += state_posteriors_at_t[i] * self.model.transmat_[i, j]
            prob_s_next[j] = sum_val
        
        # prob_s_next should sum to 1
        prob_s_next = prob_s_next / np.sum(prob_s_next) if np.sum(prob_s_next) > 1e-9 else prob_s_next


        # 3. Calculate predicted P(O_{t+1} | o_1...o_t)
        # This is the mean of the mixture distribution for the next observation.
        # For GaussianHMM, P(O_{t+1} | S_{t+1}=j) is a Gaussian.
        # The predicted observation is E[O_{t+1} | o_1...o_t] = sum_j P(S_{t+1}=j | o_1...o_t) * E[O | S_{t+1}=j]
        # E[O | S_{t+1}=j] is self.model.means_[j]
        predicted_obs_mean = np.zeros(self.model.means_.shape[1]) # D_obs
        for j in range(self.n_states):
            predicted_obs_mean += prob_s_next[j] * self.model.means_[j, :]

        # The output should be a probability distribution, so normalize it if it represents probabilities
        # The black-box output is already probabilities.
        # If predicted_obs_mean represents probabilities, ensure it sums to 1.
        # The means of Gaussian emissions are probability vectors [P(neg), P(pos)].
        if np.sum(predicted_obs_mean) > 1e-9 : # Avoid division by zero
             predicted_obs_mean_normalized = predicted_obs_mean / np.sum(predicted_obs_mean)
        else:
            # Fallback: uniform distribution if sum is zero (should not happen with positive probs)
            predicted_obs_mean_normalized = np.ones_like(predicted_obs_mean) / len(predicted_obs_mean)
            print("Warning: Predicted observation mean sums to zero. Using uniform distribution.")

        return predicted_obs_mean_normalized # This is now a normalized probability vector

    def calculate_fidelity_metrics(self, test_observation_sequences):
        """
        Calculates KL-divergence and Negative Log-Likelihood (NLL) for the HMM
        against true black-box outputs on test sequences.
        Args:
            test_observation_sequences: List of 2D numpy arrays (true black-box outputs).
        Returns:
            A dictionary with 'avg_kl_divergence' and 'avg_nll'.
        """
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")

        all_kl_divergences = []
        all_nlls = []
        total_steps_evaluated = 0

        print("Calculating fidelity metrics...")
        for true_seq in tqdm(test_observation_sequences):
            if true_seq.shape[0] < 2: # Need at least one prefix and one next step
                continue

            for t in range(true_seq.shape[0] - 1): # Iterate up to T-1 to predict for t+1
                obs_prefix = true_seq[:t+1, :]    # Observations from 0 to t
                true_next_obs = true_seq[t+1, :]  # True observation at t+1

                hmm_pred_next_dist = self.predict_next_observation_distribution(obs_prefix)

                if hmm_pred_next_dist is None:
                    continue # Skip if prediction failed for this prefix

                # Ensure no zeros in predicted distribution for KL divergence and NLL calculation
                # Add a very small epsilon to avoid log(0) or division by zero.
                epsilon = 1e-9
                hmm_pred_next_dist_clipped = np.clip(hmm_pred_next_dist, epsilon, 1.0 - epsilon)
                hmm_pred_next_dist_clipped /= np.sum(hmm_pred_next_dist_clipped) # Re-normalize

                true_next_obs_clipped = np.clip(true_next_obs, epsilon, 1.0 - epsilon)
                true_next_obs_clipped /= np.sum(true_next_obs_clipped) # Re-normalize


                # KL Divergence: D_KL(P || Q) = sum(P(x) * log(P(x)/Q(x)))
                # Here, P is true_next_obs, Q is hmm_pred_next_dist
                kl_div = np.sum(rel_entr(true_next_obs_clipped, hmm_pred_next_dist_clipped))
                all_kl_divergences.append(kl_div)

                # Negative Log-Likelihood (Cross-Entropy for probability distributions)
                # NLL = - sum(P_true(x) * log(P_pred(x)))
                nll = -np.sum(true_next_obs_clipped * np.log(hmm_pred_next_dist_clipped))
                all_nlls.append(nll)
                total_steps_evaluated +=1
        
        if not all_kl_divergences: # if no evaluations were possible
            print("Warning: No fidelity metrics could be calculated (e.g., all test sequences too short or prediction issues).")
            return {'avg_kl_divergence': float('nan'), 'avg_nll': float('nan'), 'total_steps_evaluated': 0}

        avg_kl = np.mean(all_kl_divergences)
        avg_nll = np.mean(all_nlls)

        print(f"\n--- Fidelity Metrics ({total_steps_evaluated} steps evaluated) ---")
        print(f"  Average KL Divergence (True BB || HMM Pred): {avg_kl:.4f}")
        print(f"     (Lower is better. 0 means perfect match in distribution.)")
        print(f"  Average Negative Log-Likelihood (NLL) / Cross-Entropy: {avg_nll:.4f}")
        print(f"     (Lower is better. Measures how 'surprising' the true observation is given HMM's prediction.)")

        return {'avg_kl_divergence': avg_kl, 'avg_nll': avg_nll, 'total_steps_evaluated': total_steps_evaluated}



    def calculate_detailed_fidelity_metrics(self, test_observation_sequences):
        """
        Calculates detailed KL-divergence and NLL for each individual step,
        categorizing by the true black-box output confidence level.
        Args:
            test_observation_sequences: List of 2D numpy arrays (true black-box outputs).
        Returns:
            A dictionary with detailed fidelity metrics categorized by true observation type.
        """
        if not self.is_trained:
            raise ValueError("HMM model is not trained yet.")

        # Storage for individual step metrics
        step_data = []
        total_steps_evaluated = 0

        print("Calculating detailed fidelity metrics...")
        for seq_idx, true_seq in enumerate(tqdm(test_observation_sequences)):
            if true_seq.shape[0] < 2: # Need at least one prefix and one next step
                continue

            for t in range(true_seq.shape[0] - 1): # Iterate up to T-1 to predict for t+1
                obs_prefix = true_seq[:t+1, :]    # Observations from 0 to t
                true_next_obs = true_seq[t+1, :]  # True observation at t+1

                hmm_pred_next_dist = self.predict_next_observation_distribution(obs_prefix)

                if hmm_pred_next_dist is None:
                    continue # Skip if prediction failed for this prefix

                # Ensure no zeros in predicted distribution for KL divergence and NLL calculation
                epsilon = 1e-9
                hmm_pred_next_dist_clipped = np.clip(hmm_pred_next_dist, epsilon, 1.0 - epsilon)
                hmm_pred_next_dist_clipped /= np.sum(hmm_pred_next_dist_clipped)

                true_next_obs_clipped = np.clip(true_next_obs, epsilon, 1.0 - epsilon)
                true_next_obs_clipped /= np.sum(true_next_obs_clipped)

                # Calculate metrics for this step
                kl_div = np.sum(rel_entr(true_next_obs_clipped, hmm_pred_next_dist_clipped))
                nll = -np.sum(true_next_obs_clipped * np.log(hmm_pred_next_dist_clipped))
                
                # Categorize the true observation
                # Assuming TARGET_SENTIMENT=1 (positive class index)
                true_pos_prob = true_next_obs[TARGET_SENTIMENT]
                
                if true_pos_prob <= PROB_THRESHOLDS["NEG_OBS"]:
                    category = "True Confident Negative"
                elif true_pos_prob <= PROB_THRESHOLDS["NEU_OBS"]:
                    category = "True Neutral"
                else:
                    category = "True Confident Positive"

                # Store individual step data
                step_data.append({
                    'seq_idx': seq_idx,
                    'step': t,
                    'true_obs': true_next_obs.copy(),
                    'pred_obs': hmm_pred_next_dist.copy(),
                    'true_pos_prob': true_pos_prob,
                    'category': category,
                    'kl_divergence': kl_div,
                    'nll': nll
                })
                total_steps_evaluated += 1

        if not step_data:
            print("Warning: No fidelity metrics could be calculated.")
            return None

        # Organize results by category
        results = {
            'step_data': step_data,
            'total_steps_evaluated': total_steps_evaluated,
            'by_category': {}
        }

        # Group by category and calculate statistics
        categories = ["True Confident Negative", "True Neutral", "True Confident Positive"]
        
        for category in categories:
            category_steps = [step for step in step_data if step['category'] == category]
            
            if category_steps:
                kl_values = [step['kl_divergence'] for step in category_steps]
                nll_values = [step['nll'] for step in category_steps]
                
                results['by_category'][category] = {
                    'count': len(category_steps),
                    'kl_divergences': kl_values,
                    'avg_kl': np.mean(kl_values),
                    'std_kl': np.std(kl_values),
                    'median_kl': np.median(kl_values),
                    'avg_nll': np.mean(nll_values),
                    'std_nll': np.std(nll_values),
                    'median_nll': np.median(nll_values)
                }
            else:
                results['by_category'][category] = {
                    'count': 0,
                    'kl_divergences': [],
                    'avg_kl': float('nan'),
                    'std_kl': float('nan'),
                    'median_kl': float('nan'),
                    'avg_nll': float('nan'),
                    'std_nll': float('nan'),
                    'median_nll': float('nan')
                }

        # Print detailed results
        print(f"\n--- Detailed Fidelity Analysis ({total_steps_evaluated} steps evaluated) ---")
        for category in categories:
            cat_data = results['by_category'][category]
            print(f"\n{category}:")
            print(f"  Count: {cat_data['count']}")
            if cat_data['count'] > 0:
                print(f"  KL Divergence - Mean: {cat_data['avg_kl']:.4f}, Std: {cat_data['std_kl']:.4f}, Median: {cat_data['median_kl']:.4f}")
                print(f"  NLL - Mean: {cat_data['avg_nll']:.4f}, Std: {cat_data['std_nll']:.4f}, Median: {cat_data['median_nll']:.4f}")

        # Overall statistics
        all_kl = [step['kl_divergence'] for step in step_data]
        all_nll = [step['nll'] for step in step_data]
        
        results['overall'] = {
            'avg_kl': np.mean(all_kl),
            'std_kl': np.std(all_kl),
            'avg_nll': np.mean(all_nll),
            'std_nll': np.std(all_nll)
        }

        print(f"\nOverall:")
        print(f"  KL Divergence - Mean: {results['overall']['avg_kl']:.4f}, Std: {results['overall']['std_kl']:.4f}")
        print(f"  NLL - Mean: {results['overall']['avg_nll']:.4f}, Std: {results['overall']['std_nll']:.4f}")

        return results

    def compare_hmm_performance(self, other_hmm, test_observation_sequences, model_names=("HMM 1", "HMM 2")):
        """
        Compare performance of this HMM with another HMM on the same test data.
        Args:
            other_hmm: Another HMMSurrogate instance to compare against
            test_observation_sequences: Test data to evaluate both models on
            model_names: Tuple of names for (this_model, other_model) for display
        Returns:
            Dictionary with comparative analysis
        """
        print(f"\n=== Comparing {model_names[0]} vs {model_names[1]} ===")
        
        # Get detailed metrics for both models
        results_1 = self.calculate_detailed_fidelity_metrics(test_observation_sequences)
        print(f"\n{'-'*50}")
        results_2 = other_hmm.calculate_detailed_fidelity_metrics(test_observation_sequences)
        
        if results_1 is None or results_2 is None:
            print("Cannot compare - one or both models failed to generate metrics")
            return None
        
        # Compare by category
        categories = ["True Confident Negative", "True Neutral", "True Confident Positive"]
        comparison = {
            'model_names': model_names,
            'by_category': {},
            'overall_winner': {}
        }
        
        print(f"\n=== COMPARISON SUMMARY ===")
        print(f"Format: {model_names[0]} vs {model_names[1]}")
        
        for category in categories:
            cat1 = results_1['by_category'][category]
            cat2 = results_2['by_category'][category]
            
            if cat1['count'] > 0 and cat2['count'] > 0:
                kl_diff = cat1['avg_kl'] - cat2['avg_kl']
                kl_winner = model_names[0] if kl_diff < 0 else model_names[1]
                
                comparison['by_category'][category] = {
                    'model1_avg_kl': cat1['avg_kl'],
                    'model2_avg_kl': cat2['avg_kl'],
                    'kl_difference': kl_diff,
                    'kl_winner': kl_winner,
                    'model1_count': cat1['count'],
                    'model2_count': cat2['count']
                }
                
                print(f"\n{category}:")
                print(f"  {model_names[0]}: {cat1['avg_kl']:.4f} ± {cat1['std_kl']:.4f} (n={cat1['count']})")
                print(f"  {model_names[1]}: {cat2['avg_kl']:.4f} ± {cat2['std_kl']:.4f} (n={cat2['count']})")
                print(f"  Winner: {kl_winner} (lower KL by {abs(kl_diff):.4f})")
            else:
                print(f"\n{category}: Insufficient data for comparison")
        
        # Overall comparison
        overall_kl_diff = results_1['overall']['avg_kl'] - results_2['overall']['avg_kl']
        overall_winner = model_names[0] if overall_kl_diff < 0 else model_names[1]
        
        comparison['overall_winner'] = {
            'model1_avg_kl': results_1['overall']['avg_kl'],
            'model2_avg_kl': results_2['overall']['avg_kl'],
            'kl_difference': overall_kl_diff,
            'winner': overall_winner
        }
        
        print(f"\nOVERALL:")
        print(f"  {model_names[0]}: {results_1['overall']['avg_kl']:.4f} ± {results_1['overall']['std_kl']:.4f}")
        print(f"  {model_names[1]}: {results_2['overall']['avg_kl']:.4f} ± {results_2['overall']['std_kl']:.4f}")
        print(f"  Winner: {overall_winner} (lower overall KL by {abs(overall_kl_diff):.4f})")
        
        return comparison, results_1, results_2
    
    def plot_kl_distributions(self, detailed_results, title_prefix="HMM"):
        """
        Creates plots showing KL divergence distributions by category.
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available for plotting")
            return None
        
        categories = ["True Confident Negative", "True Neutral", "True Confident Positive"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{title_prefix} - KL Divergence by True Observation Category')
        
        for i, category in enumerate(categories):
            cat_data = detailed_results['by_category'][category]
            if cat_data['count'] > 0:
                axes[i].hist(cat_data['kl_divergences'], bins=20, alpha=0.7, edgecolor='black')
                axes[i].axvline(cat_data['avg_kl'], color='red', linestyle='--', 
                              label=f'Mean: {cat_data["avg_kl"]:.3f}')
                axes[i].axvline(cat_data['median_kl'], color='orange', linestyle='--', 
                              label=f'Median: {cat_data["median_kl"]:.3f}')
                axes[i].set_title(f'{category}\n(n={cat_data["count"]})')
                axes[i].set_xlabel('KL Divergence')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{category}\n(n=0)')
        
        plt.tight_layout()
        return fig

    def analyze_prediction_disagreements(self, detailed_results, title_prefix="HMM"):
        """
        Analyze and visualize prediction disagreements using a confusion matrix heatmap.
        Shows frequency of cases where HMM prediction category differs from true observation category.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
        except ImportError:
            print("matplotlib, seaborn, or pandas not available for heatmap plotting")
            return None, None
        
        step_data = detailed_results['step_data']
        categories = ["Confident Negative", "Neutral", "Confident Positive"]
        
        # Function to categorize observations
        def categorize_observation(pos_prob):
            if pos_prob <= PROB_THRESHOLDS["NEG_OBS"]:
                return "Confident Negative"
            elif pos_prob <= PROB_THRESHOLDS["NEU_OBS"]:
                return "Neutral"
            else:
                return "Confident Positive"
        
        # Build confusion matrix
        confusion_matrix = pd.DataFrame(0, index=categories, columns=categories)
        disagreement_details = []
        total_predictions = len(step_data)
        
        print(f"\n=== PREDICTION DISAGREEMENT ANALYSIS ===")
        print(f"Analyzing {total_predictions} predictions...")
        
        for step in step_data:
            true_pos_prob = step['true_pos_prob']
            pred_pos_prob = step['pred_obs'][TARGET_SENTIMENT]  # Get predicted positive probability
            
            true_category = categorize_observation(true_pos_prob)
            pred_category = categorize_observation(pred_pos_prob)
            
            # Count in confusion matrix
            confusion_matrix.loc[true_category, pred_category] += 1
            
            # Record disagreements
            if true_category != pred_category:
                disagreement_details.append({
                    'seq_idx': step['seq_idx'],
                    'step': step['step'],
                    'true_category': true_category,
                    'pred_category': pred_category,
                    'true_pos_prob': true_pos_prob,
                    'pred_pos_prob': pred_pos_prob,
                    'kl_divergence': step['kl_divergence']
                })
        
        # Calculate disagreement statistics
        total_disagreements = len(disagreement_details)
        agreement_rate = (total_predictions - total_disagreements) / total_predictions * 100
        
        print(f"Total disagreements: {total_disagreements}/{total_predictions} ({100-agreement_rate:.1f}%)")
        print(f"Agreement rate: {agreement_rate:.1f}%")
        
        # Convert to percentages for better visualization
        confusion_percentages = confusion_matrix / total_predictions * 100
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'{title_prefix} - Prediction Counts\n(Agreement Rate: {agreement_rate:.1f}%)')
        ax1.set_xlabel('HMM Predicted Category')
        ax1.set_ylabel('True Category')
        
        # Percentage heatmap
        sns.heatmap(confusion_percentages, annot=True, fmt='.1f', cmap='Reds', 
                   ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title(f'{title_prefix} - Prediction Percentages\n(Total: {total_predictions} predictions)')
        ax2.set_xlabel('HMM Predicted Category')
        ax2.set_ylabel('True Category')
        
        plt.tight_layout()
        
        # Print detailed disagreement analysis
        if disagreement_details:
            print(f"\nDISAGREEMENT BREAKDOWN:")
            disagreement_df = pd.DataFrame(disagreement_details)
            
            # Count disagreement types
            disagreement_counts = disagreement_df.groupby(['true_category', 'pred_category']).size()
            
            print(f"\nMost common disagreement patterns:")
            for (true_cat, pred_cat), count in disagreement_counts.sort_values(ascending=False).head(5).items():
                percentage = count / total_disagreements * 100
                avg_kl = disagreement_df[(disagreement_df['true_category'] == true_cat) & 
                                       (disagreement_df['pred_category'] == pred_cat)]['kl_divergence'].mean()
                print(f"  {true_cat} → {pred_cat}: {count} cases ({percentage:.1f}% of disagreements), Avg KL: {avg_kl:.3f}")
            
            # Analyze disagreement severity by KL divergence
            high_kl_disagreements = [d for d in disagreement_details if d['kl_divergence'] > 1.0]
            print(f"\nHigh-severity disagreements (KL > 1.0): {len(high_kl_disagreements)}/{total_disagreements}")
            
            if high_kl_disagreements:
                high_kl_df = pd.DataFrame(high_kl_disagreements)
                high_kl_counts = high_kl_df.groupby(['true_category', 'pred_category']).size()
                print("High-severity disagreement patterns:")
                for (true_cat, pred_cat), count in high_kl_counts.sort_values(ascending=False).items():
                    print(f"  {true_cat} → {pred_cat}: {count} cases")
        
        # Calculate category-specific performance
        print(f"\nCATEGORY-SPECIFIC PERFORMANCE:")
        for category in categories:
            true_count = confusion_matrix.loc[category, :].sum()
            correct_count = confusion_matrix.loc[category, category]
            if true_count > 0:
                accuracy = correct_count / true_count * 100
                print(f"  {category}: {correct_count}/{true_count} correct ({accuracy:.1f}%)")
            else:
                print(f"  {category}: No samples")
        
        results = {
            'confusion_matrix': confusion_matrix,
            'confusion_percentages': confusion_percentages,
            'disagreement_details': disagreement_details,
            'agreement_rate': agreement_rate,
            'total_predictions': total_predictions,
            'total_disagreements': total_disagreements
        }
        
        return results, fig

    def compare_disagreement_patterns(self, other_hmm, test_observation_sequences, model_names=("HMM 1", "HMM 2")):
        """
        Compare disagreement patterns between two HMMs.
        """
        print(f"\n=== COMPARING DISAGREEMENT PATTERNS: {model_names[0]} vs {model_names[1]} ===")
        
        # Get detailed results for both models
        results_1 = self.calculate_detailed_fidelity_metrics(test_observation_sequences)
        results_2 = other_hmm.calculate_detailed_fidelity_metrics(test_observation_sequences)
        
        if results_1 is None or results_2 is None:
            print("Cannot compare disagreement patterns - one or both models failed")
            return None
        
        # Analyze disagreements for both models
        disagreement_1, fig_1 = self.analyze_prediction_disagreements(results_1, model_names[0])
        disagreement_2, fig_2 = other_hmm.analyze_prediction_disagreements(results_2, model_names[1])
        
        if disagreement_1 is None or disagreement_2 is None:
            print("Cannot generate disagreement analysis")
            return None
        
        # Compare agreement rates
        print(f"\n=== AGREEMENT RATE COMPARISON ===")
        print(f"{model_names[0]}: {disagreement_1['agreement_rate']:.1f}%")
        print(f"{model_names[1]}: {disagreement_2['agreement_rate']:.1f}%")
        
        better_model = model_names[0] if disagreement_1['agreement_rate'] > disagreement_2['agreement_rate'] else model_names[1]
        print(f"Better agreement: {better_model}")
        
        # Compare disagreement patterns
        print(f"\n=== DISAGREEMENT PATTERN COMPARISON ===")
        
        # Find patterns where models differ significantly
        conf1_pct = disagreement_1['confusion_percentages']
        conf2_pct = disagreement_2['confusion_percentages']
        
        categories = ["Confident Negative", "Neutral", "Confident Positive"]
        
        print(f"\nLargest differences in disagreement patterns:")
        max_diff = 0
        max_diff_pattern = None
        
        for true_cat in categories:
            for pred_cat in categories:
                if true_cat != pred_cat:  # Only look at disagreements
                    diff = abs(conf1_pct.loc[true_cat, pred_cat] - conf2_pct.loc[true_cat, pred_cat])
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_pattern = (true_cat, pred_cat)
                    
                    if diff > 1.0:  # Report differences > 1%
                        print(f"  {true_cat} → {pred_cat}:")
                        print(f"    {model_names[0]}: {conf1_pct.loc[true_cat, pred_cat]:.1f}%")
                        print(f"    {model_names[1]}: {conf2_pct.loc[true_cat, pred_cat]:.1f}%")
                        print(f"    Difference: {diff:.1f}%")
        
        if max_diff_pattern:
            true_cat, pred_cat = max_diff_pattern
            print(f"\nLargest disagreement difference: {true_cat} → {pred_cat} ({max_diff:.1f}% difference)")
        
        return {
            'model_names': model_names,
            'disagreement_1': disagreement_1,
            'disagreement_2': disagreement_2,
            'figures': (fig_1, fig_2)
        }






