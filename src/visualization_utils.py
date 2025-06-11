import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

from .config import TARGET_SENTIMENT


from .clustering_utils import log_color_shifts, save_color_log 


def print_colored_review(tokens, state_sequence, state_names=None, cmap=None, hmm_surrogate=None):
    """
    Print the review tokens in terminal with colors corresponding to HMM states.
    
    Args:
        tokens: List of string tokens
        state_sequence: Numpy array of HMM states
        state_names: Optional dictionary mapping state index to state name
        cmap: Optional colormap to use for coloring
        hmm_surrogate: Optional HMM surrogate model for enhanced state info
    """
    if len(tokens) == 0 or len(state_sequence) == 0:
        print("Empty tokens or state sequence, cannot print colored review.")
        return
    

    min_len = min(len(tokens), len(state_sequence))
    tokens = tokens[:min_len]
    state_sequence = state_sequence[:min_len]
    

    if cmap is None:
        num_states = len(np.unique(state_sequence))
        if hmm_surrogate and hasattr(hmm_surrogate, 'n_states'):
            num_states = hmm_surrogate.n_states
       
        if state_names:
            color_map_dict = {"negative": 'lightcoral', "low": 'lightcoral',
                            "neutral": 'lightgray', "uncertain": 'lightgray',
                            "positive": 'lightgreen', "high": 'lightgreen'}
            custom_colors = []
            for i in range(num_states):
                name = state_names.get(i, "").lower()
                assigned = False
                for keyword, color in color_map_dict.items():
                    if keyword in name:
                        custom_colors.append(color)
                        assigned = True
                        break
                if not assigned:
                    custom_colors.append('purple')
            cmap = mcolors.ListedColormap(custom_colors)
        else:
            cmap = plt.cm.get_cmap('viridis', num_states)
    
    def rgb_to_ansi(r, g, b):
        """Convert RGB values to ANSI escape code for terminal colors."""
        return f"\033[38;2;{int(255*r)};{int(255*g)};{int(255*b)}m"
    
    print("\n" + "="*80)
    print("REVIEW WITH HMM STATE COLORS:")
    print("="*80)

    if state_names:
        print("\nState Legend:")
        unique_states = np.unique(state_sequence)
        for state in unique_states:
            rgba = cmap(state)
            r, g, b = rgba[:3]
            ansi_color = rgb_to_ansi(r, g, b)
            state_name = state_names.get(state, f"State {state}")
            print(f"{ansi_color}■ State {state}: {state_name}\033[0m")
        print()
    

    print("Review text:")
    line_length = 0
    max_line_length = 80
    
    for i, (token, state) in enumerate(zip(tokens, state_sequence)):
        if not token.strip():
            continue
            
        rgba = cmap(state)
        r, g, b = rgba[:3]
        ansi_color = rgb_to_ansi(r, g, b)
        
        if line_length + len(token) + 1 > max_line_length and line_length > 0:
            print()
            line_length = 0
        
        print(f"{ansi_color}{token}\033[0m", end=' ')
        line_length += len(token) + 1
    
    print("\n" + "="*80 + "\n")

def plot_state_timeline(tokens, prob_trajectory, state_sequence=None, state_names=None,
                       target_class_idx=TARGET_SENTIMENT, ax=None, hmm_surrogate=None,
                       show_predictions=False):
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(18, 8))
    else:
        ax1 = ax
    fig = ax1.figure

    if state_sequence is None and hmm_surrogate is not None and hmm_surrogate.is_hmm_trained:
        try:
            state_sequence = hmm_surrogate.decode_hmm_sequence(prob_trajectory)
        except Exception as e:
            print(f"Warning: Could not decode HMM state sequence: {e}")
            return
    elif state_sequence is None:
        print("Error: state_sequence is None and no hmm_surrogate provided to decode.")
        return

    T = len(state_sequence)
    if T == 0:
        ax1.text(0.5, 0.5, "Empty state sequence, cannot plot.", ha='center', va='center')
        ax1.set_xticks([]); ax1.set_yticks([])
        if ax is None: plt.show()
        return

    min_len = min(len(prob_trajectory), T, len(tokens))
    if len(prob_trajectory) != min_len or T != min_len or len(tokens) != min_len:
        print(f"Warning: Truncating plot data to shortest available length: {min_len}")
    prob_trajectory = prob_trajectory[:min_len]
    state_sequence = state_sequence[:min_len]
    tokens = tokens[:min_len]
    T = min_len
    if T == 0:
        ax1.text(0.5, 0.5, "Sequence length became 0 after truncation.", ha='center', va='center')
        if ax is None: plt.show()
        return
        
    x_ticks = np.arange(T)

    true_target_probs = prob_trajectory[:, target_class_idx]
    ax1.plot(x_ticks, true_target_probs, label=f'True BB P(Class {target_class_idx})',
             color='dimgray', linestyle=':', alpha=0.9, linewidth=1.5)

    if show_predictions and hmm_surrogate is not None and hmm_surrogate.is_regression_trained:
        reg_pred_probs_list = []

        
        if T > 0:
            reg_pred_probs_list.append(np.nan)  

        for t_idx in range(1, T):
            obs_prefix_for_reg_features = prob_trajectory[:t_idx, :]

            single_step_processed_features = hmm_surrogate._prepare_and_transform_single_step_features(
                obs_prefix_for_reg_features
            )
            
            if single_step_processed_features is not None:
                pred_p_arr = hmm_surrogate.predict_next_step_distribution(
                    single_step_processed_features
                )
                reg_pred_probs_list.append(pred_p_arr[0] if pred_p_arr is not None else np.nan)
            else:
                reg_pred_probs_list.append(np.nan)
        
        reg_pred_probs = np.array(reg_pred_probs_list)

        if len(reg_pred_probs) == T:
            valid_reg_preds_mask = ~np.isnan(reg_pred_probs)
            if np.any(valid_reg_preds_mask):
                ax1.plot(x_ticks[valid_reg_preds_mask], reg_pred_probs[valid_reg_preds_mask],
                         label=f'Pipeline Reg. Pred P(Class {target_class_idx})',
                         color='blue', linestyle='-', linewidth=1.5)
        else:
            print(f"  Warning: Mismatch in regression prediction length. Expected {T}, got {len(reg_pred_probs)}")

    ax1.set_xlabel('Token Position', fontsize=10)
    ax1.set_ylabel(f'P(Class {target_class_idx})', fontsize=10)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9)

    ax2 = ax1.twinx()
    
    if state_names is None and hmm_surrogate is not None and hmm_surrogate.is_hmm_trained:
        try:
            if hasattr(hmm_surrogate, '_cached_hmm_state_analysis_results') and \
               hmm_surrogate._cached_hmm_state_analysis_results is not None:
                 state_names = hmm_surrogate._cached_hmm_state_analysis_results.get('state_names', {})
            else:
                 temp_obs = [prob_trajectory] if T > 0 else []
                 temp_states = [state_sequence] if T > 0 else []
                 if temp_obs and temp_states:
                     analysis_results = hmm_surrogate.analyze_hmm_states(temp_obs, temp_states, target_class_idx)
                     state_names = analysis_results.get('state_names', {})
                 else:
                     state_names = {}
        except Exception as e:
            print(f"Warning: Could not extract state names from surrogate: {e}")
            state_names = {}
    if state_names is None: state_names = {}

    unique_states_in_seq = np.unique(state_sequence)
    num_distinct_states_in_seq = len(unique_states_in_seq)
    
    num_total_hmm_states = hmm_surrogate.n_states if hmm_surrogate else (np.max(state_sequence) + 1 if T > 0 else 1)
    
    color_map_dict_semantic = {"negative": 'lightcoral', "low": 'lightcoral',
                               "neutral": 'lightgray', "uncertain": 'lightgray',
                               "positive": 'lightgreen', "high": 'lightgreen'}
    cmap_to_use = None
    custom_colors_list = ['purple'] * num_total_hmm_states
    
    semantic_colors_applied_count = 0
    if state_names:
        for state_idx in range(num_total_hmm_states):
            name = state_names.get(state_idx, "").lower()
            color_assigned = False
            for keyword, color_val in color_map_dict_semantic.items():
                if keyword in name:
                    custom_colors_list[state_idx] = color_val
                    semantic_colors_applied_count +=1
                    color_assigned = True
                    break
            if not color_assigned and state_idx < len(plt.cm.viridis.colors):
                 custom_colors_list[state_idx] = plt.cm.viridis(state_idx / max(1, num_total_hmm_states-1))

    if semantic_colors_applied_count > 0:
        cmap_to_use = mcolors.ListedColormap(custom_colors_list[:num_total_hmm_states])
    else:
        cmap_to_use = plt.cm.get_cmap('viridis', num_total_hmm_states if num_total_hmm_states > 1 else 2)

    for k_tick in range(T):
        state_val = state_sequence[k_tick]
        color_for_state = cmap_to_use(state_val if isinstance(cmap_to_use, mcolors.ListedColormap) else state_val / (num_total_hmm_states - 1e-9) )
        ax2.axvspan(k_tick - 0.5, k_tick + 0.5, color=color_for_state, alpha=0.25)

    ax2.set_yticks(unique_states_in_seq)
    ax2.set_yticklabels([state_names.get(s, f"State {s}") for s in unique_states_in_seq], fontsize=8)
    ax2.set_ylabel('Decoded HMM State', color='darkred', fontsize=10)
    if num_distinct_states_in_seq > 0:
        ax2.set_ylim(np.min(unique_states_in_seq) - 0.5, np.max(unique_states_in_seq) + 0.5)
    else:
        ax2.set_ylim(-0.5, 0.5)


    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([''] * T)

 
    text_y_position_normalized = -0.08
    if T > 0:
        for k_tick in range(T):
            state_val = state_sequence[k_tick]
            token_color = cmap_to_use(state_val if isinstance(cmap_to_use, mcolors.ListedColormap) else state_val / (num_total_hmm_states -1e-9) )
            ax1.text(x_ticks[k_tick], text_y_position_normalized, tokens[k_tick], color=token_color,
                     ha='center', va='top', rotation=90, fontsize=6, transform=ax1.get_xaxis_transform())
    
    title_str = 'BB Trajectory & HMM States'
    if show_predictions and hmm_surrogate and hmm_surrogate.is_regression_trained:
        title_str += ' w/ Pipeline Regression Pred.'
    ax1.set_title(title_str, fontsize=12)
    ax1.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    fig.tight_layout(rect=[0, 0.15, 1, 0.90])
    
 
    if T > 0:
        print_colored_review(tokens, state_sequence, state_names, cmap_to_use, hmm_surrogate)
        
       
        if 'log_color_shifts' in globals():
            try:
                log_color_shifts(tokens, state_sequence, cmap_to_use)
            except Exception as e_log:
                print(f"Note: Could not log color shifts: {e_log}")

    if ax is None:
        plt.show()

def plot_hmm_transition_matrix(hmm_model, state_names=None):
    if not hasattr(hmm_model, 'transmat_'):
        print("HMM model does not have a transition matrix (transmat_). Is it trained?")
        return
    transmat = hmm_model.transmat_
    num_states_in_model = transmat.shape[0]
    
    fig_width = max(8, num_states_in_model * 2.0) 
    fig_height = max(6, num_states_in_model * 1.8) 
    
   
    plt.figure(figsize=(fig_width, fig_height))
    
  
    tick_labels = [state_names.get(i, f"State {i}") for i in range(num_states_in_model)] if state_names else [f"State {i}" for i in range(num_states_in_model)]
    
 
    ax = sns.heatmap(transmat, 
                     annot=True, 
                     fmt=".2f", 
                     cmap="viridis",
                     xticklabels=tick_labels, 
                     yticklabels=tick_labels,
                     annot_kws={"size": 10},  
                     cbar_kws={'label': 'Transition Probability'})
    
    
    plt.xlabel("To State", fontsize=12, labelpad=10)
    plt.ylabel("From State", fontsize=12, labelpad=10)
    plt.title("HMM State Transition Matrix", fontsize=14, pad=20)
    

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
   
    plt.tight_layout(pad=2.0)
    
    plt.show()

def plot_avg_probabilities_per_state(state_analysis_results, target_class_idx=TARGET_SENTIMENT):
    if not state_analysis_results or 'state_names' not in state_analysis_results:
        print("State analysis results incomplete. Cannot plot average probabilities.")
        return
    state_names = state_analysis_results['state_names']
    state_indices = sorted([k for k in state_analysis_results if isinstance(k, int)])
    if not state_indices:
        print("No state data found in state_analysis_results.")
        return

    avg_probs = [state_analysis_results[s]['avg_prob_target_class'] for s in state_indices]
    labels = [state_names.get(s, f"State {s}") for s in state_indices]

    fig, ax_bar = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bars = ax_bar.bar(labels, avg_probs, color='skyblue', label='Avg P(Target Class)')
    ax_bar.set_xlabel("HMM State")
    ax_bar.set_ylabel(f"Average P(Class {target_class_idx})", color='skyblue')
    ax_bar.tick_params(axis='y', labelcolor='skyblue')
    plt.xticks(rotation=45, ha="right")
    ax_bar.set_ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.title(f"HMM State Characteristics (Target Class {target_class_idx})", y=1.12)
    plt.tight_layout(rect=[0,0,1,0.9])
    plt.show()

def plot_state_timeline_from_surrogate(hmm_surrogate, tokens, prob_trajectory, 
                                     target_class_idx=TARGET_SENTIMENT, ax=None, 
                                     show_predictions=True):
    if not hmm_surrogate or not hmm_surrogate.is_hmm_trained:
        print("Error: HMM surrogate instance not provided or not trained.")
        return
    
    state_names_for_plot = {}
    if hasattr(hmm_surrogate, '_cached_hmm_state_analysis_results') and \
       hmm_surrogate._cached_hmm_state_analysis_results is not None:
        state_names_for_plot = hmm_surrogate._cached_hmm_state_analysis_results.get('state_names', {})
    elif prob_trajectory.shape[0] > 0:
        try:
            temp_state_seq = hmm_surrogate.decode_hmm_sequence(prob_trajectory)
            if temp_state_seq.size > 0:
                analysis_results = hmm_surrogate.analyze_hmm_states([prob_trajectory], [temp_state_seq], target_class_idx)
                state_names_for_plot = analysis_results.get('state_names', {})
        except Exception:
            pass

    return plot_state_timeline(
        tokens=tokens,
        prob_trajectory=prob_trajectory,
        state_sequence=None,  
        state_names=state_names_for_plot,
        target_class_idx=target_class_idx,
        ax=ax,
        hmm_surrogate=hmm_surrogate,
        show_predictions=show_predictions
    )