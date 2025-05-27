import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import TARGET_SENTIMENT

def plot_state_timeline(tokens, prob_trajectory, state_sequence, state_names=None, target_class_idx=TARGET_SENTIMENT, ax=None):
    """
    Plots the probability trajectory and the HMM state timeline for a single review.
    Args:
        tokens: List of string tokens for the review.
        prob_trajectory: Numpy array (T, C) of probabilities.
        state_sequence: Numpy array (T,) of HMM states.
        state_names: Optional dictionary mapping state index to state name.
        target_class_idx: Index of the probability to plot (e.g., P(positive)).
        ax: Optional matplotlib Axes object.
    """
    if ax is None:
        # Increased figure height to accommodate the sentence better
        fig, ax1 = plt.subplots(figsize=(15, 7.5)) # Original: (15,6), adjusted for sentence
    else:
        ax1 = ax
    
    fig = ax1.figure # Get the figure object

    T = len(state_sequence)
    if T == 0:
        ax1.text(0.5, 0.5, "Empty sequence, cannot plot.", ha='center', va='center')
        ax1.set_xticks([])
        ax1.set_yticks([])
        if ax is None: # Only show if we created the plot
            plt.show()
        return

    x_ticks = np.arange(T)

    # Plot probability of target class
    target_probs = prob_trajectory[:, target_class_idx]
    ax1.plot(x_ticks, target_probs, label=f'P(Class {target_class_idx})', color='black', linestyle='--')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel(f'P(Class {target_class_idx})', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 1)

    # Create a second y-axis for HMM states (colored background)
    ax2 = ax1.twinx()
    
    unique_states_arr = np.unique(state_sequence)
    num_distinct_states = len(unique_states_arr) if len(unique_states_arr) > 0 else 1
    max_state_val = 0
    if len(unique_states_arr) > 0:
        max_state_val = np.max(unique_states_arr)
    
    # Number of entries the colormap should have to cover all state values up to max_state_val
    colormap_num_entries = max_state_val + 1
    
    # Determine colormap
    color_map_dict_semantic = {
        "negative": 'red',
        "low": 'red',
        "neutral": 'lightgray',
        "uncertain": 'lightgray',
        "positive": 'green',
        "high": 'green'
    }
    custom_colors_list = []
    use_custom_cmap = False

    if state_names:
        # Check if state_names suggest a semantic coloring (neg, neu, pos)
        # This covers the 4-state merged to 3 semantic categories case as well
        semantic_keys_found = set()
        temp_color_assignment = {}

        # Determine the number of entries needed for the custom colormap
        # It should be large enough to cover all keys in state_names and all observed states.
        max_s_idx_observed = np.max(unique_states_arr) if len(unique_states_arr) > 0 else -1
        max_s_name_idx = max(state_names.keys()) if state_names else -1
        num_custom_cmap_entries = max(max_s_idx_observed, max_s_name_idx) + 1

        for state_idx in range(num_custom_cmap_entries):
            name = state_names.get(state_idx, f"State {state_idx}") # Get name or default
            name_lower = name.lower()
            assigned_color = 'purple' # Default color for unmapped states
            for keyword, color_val in color_map_dict_semantic.items():
                if keyword in name_lower:
                    assigned_color = color_val
                    semantic_keys_found.add(keyword.split('/')[0]) # e.g., add 'negative' from "negative/low"
                    break
            temp_color_assignment[state_idx] = assigned_color
        
        # Use custom map if at least two distinct semantic colors were assigned
        # (e.g., we found something for negative and something for positive)
        if len(set(temp_color_assignment.values())) > 1 and len(semantic_keys_found) >=2:
            use_custom_cmap = True
            custom_colors_list = [temp_color_assignment.get(i, 'purple') for i in range(num_custom_cmap_entries)]
            cmap = mcolors.ListedColormap(custom_colors_list)
        
    if not use_custom_cmap:
        # Fallback to default colormap if custom conditions aren't met
        cmap = plt.cm.get_cmap('viridis', colormap_num_entries)

    for i in range(T):
        state = state_sequence[i]
        # Colormaps created with a specific number of entries (lut) or ListedColormap expect integer indices.
        ax2.axvspan(i - 0.5, i + 0.5, color=cmap(state), alpha=0.3)

    # Set y-ticks and labels for HMM states
    # If states are sparse (e.g., 0, 2, 5), use unique_states_arr for ticks.
    ax2.set_yticks(unique_states_arr) 
    if state_names:
        ax2.set_yticklabels([state_names.get(s, f"State {s}") for s in unique_states_arr])
    else:
        ax2.set_yticklabels([f"State {s}" for s in unique_states_arr])
    ax2.set_ylabel('HMM State', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    # Adjust y-limits based on actual state values
    if len(unique_states_arr) > 0:
        ax2.set_ylim(np.min(unique_states_arr) - 0.5, np.max(unique_states_arr) + 0.5)
    else: # Should not happen due to T=0 check, but as a fallback
        ax2.set_ylim(-0.5, 0.5)


    ax1.set_xticks(x_ticks)
    
    if T > 0:
        text_y_position = -0.05 
        ax1.set_xticklabels([''] * T) # Hide default numerical x-axis tick labels

        for i in range(T):
            state = state_sequence[i]
            token_color = cmap(state) # Use integer state index
            ax1.text(x_ticks[i], text_y_position, tokens[i], color=token_color,
                     ha='center', va='top', rotation=90, fontsize=8, transform=ax1.get_xaxis_transform())
    
    ax1.set_title('Black-Box Probability Trajectory and HMM State Timeline')
    ax1.grid(True, axis='x', linestyle=':', alpha=0.7)
    
    # Adjust layout to make space for the sentence at the bottom
    fig.tight_layout(rect=[0, 0.15, 1, 0.95]) # Increased bottom margin for sentence

    # --- MODIFIED SECTION: Add colored sentence string ---
    if T > 0:
        # Ensure renderer is available. This might require the figure to be drawn once.
        # For most backends, it's available after plt.subplots().
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            # Fallback if renderer is not immediately available (e.g. some backends or contexts)
            # This might lead to less accurate text placement if the fallback width calculations are used.
            print("Warning: Could not get renderer immediately. Text layout for sentence might be suboptimal.")
            renderer = None 

        # Parameters for the colored sentence string
        initial_sentence_y_fig = 0.10 # Y position in figure coordinates (fraction from bottom)
        current_sentence_y_fig = initial_sentence_y_fig
        margin_x_fig = 0.03  # Starting X position (left margin)
        current_x_fig = margin_x_fig
        sentence_fontsize = 10 # Increased font size for readability
        line_spacing_factor = 1.5 # Multiplier for font size to get line height
        
        line_height_points = sentence_fontsize * line_spacing_factor
        line_height_fig = line_height_points / (fig.get_figheight() * fig.dpi)

        # Calculate width of a standard space character in figure coordinates
        space_width_fig = (sentence_fontsize * 0.3) / (fig.get_figwidth() * fig.dpi) # Fallback
        if renderer:
            try:
                space_text_obj = fig.text(0, 0, " ", fontsize=sentence_fontsize, visible=False, transform=fig.transFigure)
                space_bbox = space_text_obj.get_window_extent(renderer=renderer)
                space_width_fig = space_bbox.width / (fig.get_figwidth() * fig.dpi)
                space_text_obj.remove()
            except Exception as e:
                print(f"Warning: Failed to calculate space width accurately: {e}")


        max_x_fig = 0.97 # Don't let text go beyond 97% of figure width (right margin)
        min_y_fig_for_ellipsis = 0.01 # If text goes below this y-coordinate, show ellipsis

        for i in range(T):
            token_str = tokens[i]
            state = state_sequence[i]
            token_color = cmap(state)

            # Calculate actual token width in figure coordinates using a temporary text object
            token_width_fig = (len(token_str) * sentence_fontsize * 0.55) / (fig.get_figwidth() * fig.dpi) # Fallback
            if renderer:
                try:
                    temp_text_obj = fig.text(0, 0, token_str, fontsize=sentence_fontsize, visible=False, transform=fig.transFigure)
                    token_bbox = temp_text_obj.get_window_extent(renderer=renderer)
                    token_width_fig = token_bbox.width / (fig.get_figwidth() * fig.dpi)
                    temp_text_obj.remove()
                except Exception as e:
                     print(f"Warning: Failed to calculate token width for '{token_str}': {e}")


            # If adding this token would overflow current line (and it's not the first token on the line)
            if current_x_fig + token_width_fig > max_x_fig and current_x_fig > margin_x_fig + 1e-6: # Epsilon for float comparison
                current_x_fig = margin_x_fig # Reset to left margin
                current_sentence_y_fig -= line_height_fig # Move to a new line below

            # Safety break if text goes too far down (multiple lines make y too small)
            # Check if not on the first line to avoid breaking if initial_sentence_y_fig is already too low
            if current_sentence_y_fig < min_y_fig_for_ellipsis and \
               abs(current_sentence_y_fig - initial_sentence_y_fig) > 1e-6 : 
                fig.text(current_x_fig, current_sentence_y_fig, "...", 
                         transform=fig.transFigure, fontsize=sentence_fontsize, color='black',
                         ha='left', va='bottom')
                break 
            
            fig.text(current_x_fig, current_sentence_y_fig, token_str,
                     color=token_color,
                     fontsize=sentence_fontsize,
                     ha='left',
                     va='bottom', # Anchor text from its bottom-left
                     transform=fig.transFigure)
            
            current_x_fig += token_width_fig + space_width_fig
    # --- END MODIFIED SECTION ---

    if ax is None:
        plt.show()


def plot_hmm_transition_matrix(hmm_model, state_names=None):
    """Plots the HMM transition matrix as a heatmap."""
    if not hasattr(hmm_model, 'transmat_'):
        print("HMM model does not have a transition matrix (transmat_). Is it trained?")
        return

    transmat = hmm_model.transmat_
    num_states_in_model = transmat.shape[0] # This is n_components of HMM

    plt.figure(figsize=(8, 6))
    
    # Tick labels should correspond to 0..N-1 where N is n_components
    tick_labels = [state_names.get(i, f"State {i}") for i in range(num_states_in_model)] if state_names else [f"State {i}" for i in range(num_states_in_model)]

    sns.heatmap(transmat, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("HMM State Transition Matrix")
    plt.show()

def plot_avg_probabilities_per_state(state_analysis_results, target_class_idx=TARGET_SENTIMENT):
    """Plots the average probability of the target class for each HMM state."""
    if 'state_names' not in state_analysis_results:
        print("State analysis results incomplete. Cannot plot average probabilities.")
        return
        
    state_names = state_analysis_results['state_names']
    # Assuming state_analysis_results keys are state indices 0, 1, 2...
    # And that state_names maps these indices to names.
    # The number of bars will be len(state_names) if state_names covers all relevant states.
    # Or, more robustly, iterate through sorted keys of state_analysis_results (excluding 'state_names')
    
    state_indices = sorted([k for k in state_analysis_results if isinstance(k, int)])
    if not state_indices:
        print("No state data found in state_analysis_results.")
        return

    avg_probs = [state_analysis_results[s]['avg_prob_target_class'] for s in state_indices]
    labels = [state_names.get(s, f"State {s}") for s in state_indices]

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5)) # Adjust width based on number of states
    bars = plt.bar(labels, avg_probs, color='skyblue')
    plt.xlabel("HMM State")
    plt.ylabel(f"Average P(Class {target_class_idx})")
    plt.title(f"Average Probability of Target Class {target_class_idx} per HMM State")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Example Usage for plot_state_timeline ---
    # Using a longer sentence to test wrapping
    dummy_tokens = ("This is a much longer sentence to test the wrapping capabilities "
                    "of the revised plotting function, hopefully it works well and "
                    "breaks lines appropriately without overlapping text. "
                    "Let's add even more words to ensure multiple lines are tested "
                    "effectively, especially with varying token lengths like 'a' vs 'appropriately'.").split()
    
    num_dummy_tokens = len(dummy_tokens)
    # Generate dummy probabilities and states for the longer sentence
    np.random.seed(42) # for reproducibility
    dummy_probs = np.random.rand(num_dummy_tokens, 2)
    dummy_probs = dummy_probs / np.sum(dummy_probs, axis=1, keepdims=True) # Normalize to sum to 1

    dummy_states = np.random.randint(0, 3, size=num_dummy_tokens) # States 0, 1, 2
    
    dummy_state_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

    plot_state_timeline(dummy_tokens, dummy_probs, dummy_states, state_names=dummy_state_names, target_class_idx=1)

    # --- Example Usage for plot_hmm_transition_matrix (requires a trained HMM model) ---
    from hmmlearn import hmm
    # Note: hmmlearn API might change. For newer versions, n_components, n_iter might need specific init_params.
    try:
        dummy_hmm = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=10, init_params="stmc")
        dummy_hmm.means_ = np.random.rand(3,2) # dummy means
        dummy_hmm.covars_ = np.tile(np.eye(2), (3,1,1)) # dummy covars
        dummy_hmm.startprob_ = np.array([0.4,0.3,0.3]) # dummy start probs
        dummy_hmm.transmat_ = np.array([[0.7,0.2,0.1],[0.1,0.6,0.3],[0.2,0.3,0.5]]) # dummy transmat
        # X_dummy = np.random.rand(100, 2) 
        # lengths_dummy = [50, 50]
        # dummy_hmm.fit(X_dummy, lengths_dummy) # Fitting might be complex to set up for a minimal example
        print("Using pre-defined HMM parameters for transition matrix plot.")
    except Exception as e: # Fallback for older hmmlearn or other issues
        print(f"Could not initialize GaussianHMM with full params, trying basic: {e}")
        dummy_hmm = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=10)
        X_dummy = np.random.rand(100, 2) 
        lengths_dummy = [50, 50]
        dummy_hmm.fit(X_dummy, lengths_dummy)

    plot_hmm_transition_matrix(dummy_hmm, state_names=dummy_state_names)

    # --- Example Usage for plot_avg_probabilities_per_state ---
    dummy_state_analysis = {
        0: {'avg_prob_target_class': 0.15, 'num_occurrences': 100},
        1: {'avg_prob_target_class': 0.52, 'num_occurrences': 80},
        2: {'avg_prob_target_class': 0.88, 'num_occurrences': 120},
        'state_names': dummy_state_names
    }
    plot_avg_probabilities_per_state(dummy_state_analysis, target_class_idx=1)