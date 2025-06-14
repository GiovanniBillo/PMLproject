import numpy as np
from hmmlearn import hmm
import joblib
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from .config import NUM_HMM_STATES, HMM_N_ITER, HMM_TOL, HMM_COV_TYPE, \
                       TARGET_SENTIMENT, HMM_MODEL_PATH, PROB_THRESHOLDS

class HMMRegressionSurrogate:
    def __init__(self, n_states=NUM_HMM_STATES, n_iter=HMM_N_ITER, tol=HMM_TOL,
                 covariance_type=HMM_COV_TYPE, random_state=2,
                 regression_method='random_forest',
                 min_hmm_covar=1e-3, 
                 use_feature_scaling=True, use_pca_features=False, n_pca_components=10,
                 covar_floor_factor=1e-4, dirichlet_alpha=0.1):
        
        self.n_states = n_states
        self.regression_method = regression_method
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.target_sentiment_idx = TARGET_SENTIMENT

        self.use_feature_scaling = use_feature_scaling
        self.feature_scaler = StandardScaler() if use_feature_scaling else None
        
        self.use_pca_features = use_pca_features
        self.n_pca_components = n_pca_components
        self.pca_transformer = PCA(n_components=self.n_pca_components, random_state=random_state) if use_pca_features else None

        self.covar_floor_factor = covar_floor_factor
        self.dirichlet_alpha = dirichlet_alpha
        self.adaptive_min_covar = max(min_hmm_covar, covar_floor_factor)

        self.hmm_model = hmm.GaussianHMM(
            n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, tol=tol,
            random_state=random_state, verbose=True, min_covar=self.adaptive_min_covar)

        self._init_regression_models()
        
        self.final_sentiment_classifier = MLPClassifier(
            random_state=self.random_state + 200, 
            early_stopping=True, 
            max_iter=300,
            hidden_layer_sizes=(64, 32)
        )
        self.final_pred_feature_scaler = StandardScaler()
        self.use_final_pred_scaling = True
        
        self.is_hmm_trained = False
        self.is_regression_trained = False
        self.is_final_classifier_trained = False
        self.feature_names = []

    

    def _apply_dirichlet_smoothing(self):
        """Apply Dirichlet smoothing to transition matrix after training."""
        if not hasattr(self.hmm_model, 'transmat_'):
            return
            
        smoothed_transmat = self.hmm_model.transmat_ + self.dirichlet_alpha
        row_sums = np.sum(smoothed_transmat, axis=1, keepdims=True)
        smoothed_transmat = smoothed_transmat / row_sums
        
        print(f"Applied Dirichlet smoothing (alpha={self.dirichlet_alpha}) to transition matrix")
        print("Original min transition prob:", np.min(self.hmm_model.transmat_))
        print("Smoothed min transition prob:", np.min(smoothed_transmat))
        
        self.hmm_model.transmat_ = smoothed_transmat

    def _apply_covariance_flooring(self, observation_data_concatenated): 
        """
        Apply adaptive covariance flooring based on observation data characteristics.
        Sets self.adaptive_covar_floor (shape (n_dim,))
        and updates self.hmm_model.min_covar (scalar) for hmmlearn.
        """
        if observation_data_concatenated.shape[0] < 2:
            print("Warning: Not enough data to compute observation variance for adaptive flooring. Using default min_covar.")
            
            self.adaptive_covar_floor = np.full(observation_data_concatenated.shape[1], self.adaptive_min_covar)
            
            return self.adaptive_covar_floor

        obs_var = np.var(observation_data_concatenated, axis=0)
        
        
        self.adaptive_covar_floor = np.maximum(
            obs_var * self.covar_floor_factor,
            np.full_like(obs_var, self.adaptive_min_covar)
        )
        
     
        scalar_min_covar_for_hmmlearn = max(np.mean(self.adaptive_covar_floor), self.adaptive_min_covar)
        self.hmm_model.min_covar = scalar_min_covar_for_hmmlearn
            
        print(f"Adaptive covariance floor calculated (per dim): {self.adaptive_covar_floor}")
        print(f"Set scalar min_covar for hmmlearn during fit: {self.hmm_model.min_covar}")
        return self.adaptive_covar_floor

    def _post_process_hmm_parameters(self):
        """Post-process HMM parameters for numerical stability."""
        if not self.is_hmm_trained or not hasattr(self.hmm_model, 'covars_'):
            print("HMM not trained or covars not available for post-processing.")
            return

        n_dim_obs = self.hmm_model.means_.shape[1] 

        if self.covariance_type == 'diag':
            current_covars = self.hmm_model.covars_.copy() 

            if current_covars.shape != (self.n_states, n_dim_obs):
                print(f"ERROR: Initial covars_ shape {current_covars.shape} is unexpected for 'diag'. "
                      f"Expected ({self.n_states}, {n_dim_obs}). Aborting post-process.")
                return 

            if hasattr(self, 'adaptive_covar_floor') and self.adaptive_covar_floor.shape == (n_dim_obs,):
                cov_floor = self.adaptive_covar_floor 
            else:
                cov_floor = self.hmm_model.min_covar 
            
            processed_covars = np.maximum(current_covars, cov_floor)

            if processed_covars.shape == (self.n_states, n_dim_obs):
                self.hmm_model.covars_ = processed_covars
                print(f"Post-processed 'diag' covariances. Final shape: {self.hmm_model.covars_.shape}")
            else:
                print(f"ERROR: Processed 'diag' covars have incorrect shape {processed_covars.shape}. "
                      f"Expected ({self.n_states}, {n_dim_obs}). Covariances not updated.")

        elif self.covariance_type == 'full':
            for i in range(self.n_states):
                if hasattr(self, 'adaptive_covar_floor') and self.adaptive_covar_floor.shape == (n_dim_obs,):
                    reg_value = np.mean(self.adaptive_covar_floor) 
                else:
                    reg_value = self.hmm_model.min_covar 
                
                diag_indices = np.diag_indices(n_dim_obs)
                self.hmm_model.covars_[i][diag_indices] = np.maximum(
                    self.hmm_model.covars_[i][diag_indices], reg_value
                )
            print("Post-processed 'full' covariances (diagonal floored).")
            
        elif self.covariance_type == 'spherical':
            if hasattr(self, 'adaptive_covar_floor'):
                floor_val = np.mean(self.adaptive_covar_floor) 
            else:
                floor_val = self.hmm_model.min_covar
            self.hmm_model.covars_ = np.maximum(self.hmm_model.covars_, floor_val)
            print("Post-processed 'spherical' covariances.")

        elif self.covariance_type == 'tied':
            if hasattr(self, 'adaptive_covar_floor') and self.adaptive_covar_floor.shape == (n_dim_obs,):
                diag_indices = np.diag_indices(n_dim_obs)
                self.hmm_model.covars_[diag_indices] = np.maximum(
                    self.hmm_model.covars_[diag_indices], self.adaptive_covar_floor
                )
            else: 
                diag_indices = np.diag_indices(n_dim_obs)
                self.hmm_model.covars_[diag_indices] = np.maximum(
                    self.hmm_model.covars_[diag_indices], self.hmm_model.min_covar
                )
            print("Post-processed 'tied' covariances (diagonal floored).")


        if hasattr(self.hmm_model, 'startprob_'):
            min_start_prob = 1e-6 
            self.hmm_model.startprob_ = np.maximum(self.hmm_model.startprob_, min_start_prob)
            self.hmm_model.startprob_ /= np.sum(self.hmm_model.startprob_)

    def fit_hmm(self, observation_sequences):
        """Train the HMM component with enhanced robustness features"""
        if not observation_sequences: 
            return False
        lengths = [seq.shape[0] for seq in observation_sequences if seq.shape[0] > 0] 
        if not lengths: 
            return False
        concatenated_sequences = np.concatenate([seq for seq in observation_sequences if seq.shape[0] > 0])

        print("\n--- Applying HMM Robustness Enhancements (Pre-Fit) ---")
        self._apply_covariance_flooring(concatenated_sequences) 
        
        print(f"\nTraining HMM with {self.n_states} states (cov_type='{self.covariance_type}') on {len(lengths)} sequences.")
        print(f"Using min_covar for hmmlearn fit: {self.hmm_model.min_covar}")
        
        original_verbose_setting = self.hmm_model.verbose
        self.hmm_model.verbose = True 
        try:
            self.hmm_model.fit(concatenated_sequences, lengths)
        finally:
            self.hmm_model.verbose = original_verbose_setting 

        # Post-fit processing
        print("\n--- Applying HMM Robustness Enhancements (Post-Fit) ---")
        self._apply_dirichlet_smoothing()    
        self._post_process_hmm_parameters()  

        self.is_hmm_trained = True
        print("\nHMM training complete with robustness enhancements.")
        self.print_hmm_parameters() 
        self._define_feature_schema()
        return True

    def print_hmm_parameters(self):
        print("--- Enhanced HMM Parameters ---")
        print("Initial state probabilities (startprob_):\n", self.hmm_model.startprob_)
        print("\nTransition matrix (transmat_):\n", self.hmm_model.transmat_)
        print("\nEmission probabilities (means_ for GaussianHMM):\n", self.hmm_model.means_)
        if hasattr(self.hmm_model, 'covars_'):
            print("\nEmission covariances (covars_ for GaussianHMM):\n", self.hmm_model.covars_)
            print(f"Covariance floor applied: {self.hmm_model.min_covar}")
        else:
            print("\nEmission covariances (covars_ for GaussianHMM): Not available")
        
        print(f"\n--- Robustness Metrics ---")
        print(f"Minimum transition probability: {np.min(self.hmm_model.transmat_):.6f}")
        print(f"Maximum transition probability: {np.max(self.hmm_model.transmat_):.6f}")
        print(f"Minimum start probability: {np.min(self.hmm_model.startprob_):.6f}")
        if hasattr(self.hmm_model, 'covars_'):
            if self.covariance_type == 'diag':
                print(f"Minimum diagonal covariance: {np.min(self.hmm_model.covars_):.6f}")
            print(f"Dirichlet smoothing alpha: {self.dirichlet_alpha}")


        
       

    def _init_regression_models(self):
        if self.regression_method == 'random_forest':
            self.prob_regressor = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)
            self.lower_quantile_regressor = GradientBoostingRegressor(
                loss='quantile', alpha=0.025,
                n_estimators=50, max_depth=8, random_state=self.random_state + 1) 
            self.upper_quantile_regressor = GradientBoostingRegressor(
                loss='quantile', alpha=0.975,
                n_estimators=50, max_depth=8, random_state=self.random_state + 2)

        elif self.regression_method == 'mlp':
            self.prob_regressor = MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                alpha=0.01, random_state=self.random_state, max_iter=500, early_stopping=True, n_iter_no_change=10)

            self.lower_quantile_regressor = None 
            self.upper_quantile_regressor = None

        elif self.regression_method == 'ensemble':
            self.prob_regressors = [
                RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1),
                MLPRegressor(hidden_layer_sizes=(64,32), random_state=self.random_state + 1, max_iter=300, early_stopping=True),
                RandomForestRegressor(n_estimators=50, max_depth=15, random_state=self.random_state + 2, n_jobs=-1)
            ]
            self.lower_quantile_regressor = None
            self.upper_quantile_regressor = None
        else:
            raise ValueError(f"Unknown regression_method: {self.regression_method}")

   

    def fit_regression(self, X_train_processed, y_train, X_val_processed=None, y_val=None):
        """Fits the regression models on pre-extracted, pre-scaled, and pre-PCA-transformed features."""
        if X_train_processed.shape[0] == 0:
            print("Error: No training features for regression.")
            return False

        print(f"Fitting regression models on {X_train_processed.shape[0]} processed training samples.")
        if self.regression_method == 'ensemble':
            for i, regressor in enumerate(self.prob_regressors):
                print(f"Fitting ensemble probability model {i+1}/{len(self.prob_regressors)}")
                regressor.fit(X_train_processed, y_train)
        else:
            print("Fitting probability regressor...")
            self.prob_regressor.fit(X_train_processed, y_train)
            if self.regression_method == 'random_forest' and self.lower_quantile_regressor and self.upper_quantile_regressor:
                print("Fitting quantile regressors for prediction intervals...")
                self.lower_quantile_regressor.fit(X_train_processed, y_train)
                self.upper_quantile_regressor.fit(X_train_processed, y_train)
        
        self.is_regression_trained = True
        print("Regression model(s) fitting complete.")

        if X_val_processed is not None and y_val is not None and X_val_processed.shape[0] > 0:
            self._validate_regression(X_val_processed, y_val)
        else:
            print("No validation data provided or validation set empty for immediate regression validation.")
        return True

    def _validate_regression(self, X_val_processed, y_val):
        """Validate regression performance on processed (scaled/PCA'd) validation features."""
        if not self.is_regression_trained or X_val_processed.shape[0] == 0:
            return
            
        pred_probs = self.predict_next_step_distribution(X_val_processed)
        
        prob_mse = mean_squared_error(y_val, pred_probs)
        prob_mae = mean_absolute_error(y_val, pred_probs)
        
        print(f"\n--- Regression Validation Results ---")
        print(f"  Target Probability Prediction - MSE: {prob_mse:.4f}, MAE: {prob_mae:.4f}")
        print(f"     Predicted Probs Range: [{pred_probs.min():.3f}-{pred_probs.max():.3f}], True Probs Range: [{y_val.min():.3f}-{y_val.max():.3f}]")

    def predict_next_step_distribution(self, processed_features_matrix):
        """
        Predicts P(TARGET_SENTIMENT) for the next step.
        Args:
            processed_features_matrix: Numpy array of features for the current step(s) (N_samples, N_processed_features).
                                       These features MUST ALREADY BE SCALED AND PCA-TRANSFORMED
                                       if scaling and PCA were used during training.
        """
        if not self.is_regression_trained:
            raise ValueError("Regression models must be trained first.")
        
        features_matrix_final = np.atleast_2d(processed_features_matrix)
        
        if self.regression_method == 'ensemble':
            ensemble_preds = np.array([reg.predict(features_matrix_final) for reg in self.prob_regressors])
            pred_probs = np.mean(ensemble_preds, axis=0)
        elif self.regression_method == 'random_forest':
            pred_probs = self.prob_regressor.predict(features_matrix_final)
        elif self.regression_method == 'mlp':
            if features_matrix_final.shape[1] != self.prob_regressor.n_features_in_:
                raise ValueError(
                    f"MLP Predict: Input features have {features_matrix_final.shape[1]} dimensions, "
                    f"but MLP regressor expects {self.prob_regressor.n_features_in_}."
                )
            pred_probs = self.prob_regressor.predict(features_matrix_final)
            
        return pred_probs

    def _prepare_and_transform_single_step_features(self, obs_prefix_sequence):
        """
        Prepares raw features for a single step and then applies fitted scaler and PCA.
        This is the method that should be called by external inference loops (like visualization).
        """
        raw_features = self._prepare_single_step_features_for_prediction(obs_prefix_sequence)
        if raw_features is None:
            return None

        processed_features = raw_features
        if self.feature_scaler and hasattr(self.feature_scaler, 'mean_'):
            if processed_features.shape[1] != self.feature_scaler.n_features_in_:
                 print(f"Warning: Feature mismatch for scaling in prediction. Raw: {processed_features.shape[1]}, Scaler expects: {self.feature_scaler.n_features_in_}")
                 return None
            processed_features = self.feature_scaler.transform(processed_features)
        
        if self.pca_transformer and hasattr(self.pca_transformer, 'mean_'):
            if processed_features.shape[1] != self.pca_transformer.n_features_in_:
                print(f"Warning: Feature mismatch for PCA in prediction. Scaled: {processed_features.shape[1]}, PCA expects: {self.pca_transformer.n_features_in_}")
                return None
            processed_features = self.pca_transformer.transform(processed_features)
        
        return processed_features

    
    def _define_feature_schema(self):
        """Defines the names and order of features for consistency."""
        self.feature_names = []
        self.feature_names.extend([f'state_post_{j}' for j in range(self.n_states)])
        self.feature_names.extend([f'next_hmm_state_prob_{j}' for j in range(self.n_states)])
        self.feature_names.append('decoded_current_state')
        self.feature_names.append('max_state_posterior_confidence')
        self.feature_names.append('norm_seq_position')
        
        print(f"Defined HMM-only feature schema with {len(self.feature_names)} features.")

    def _extract_regression_features(self, observation_sequences, for_training=True):
        """Extract features for regression. If for_training=True, also extracts targets."""
        if not self.is_hmm_trained:
            raise ValueError("HMM must be trained first to extract features.")
        if not self.feature_names:
            self._define_feature_schema()

        all_features, all_targets = [], []
        d_obs = self.hmm_model.means_.shape[1]

        for seq in tqdm(observation_sequences, desc="Extracting regression features"):
            if seq.shape[0] < (2 if for_training else 1): continue
            
            try:
                state_posteriors = self.hmm_model.predict_proba(seq)
                decoded_states = self.hmm_model.predict(seq)
            except Exception as e:
                continue

            loop_range = len(seq) - 1 if for_training else len(seq)

            for t in range(loop_range):
                current_obs = seq[t]
                if current_obs.shape[0] != d_obs:
                    continue

                current_state_post = state_posteriors[t]
                current_decoded_state = decoded_states[t]
                next_hmm_state_probs = current_state_post @ self.hmm_model.transmat_
                norm_seq_pos = t / (len(seq) -1) if len(seq) > 1 else 0.0

                
                feature_vector_list = [
                    current_state_post,          
                    next_hmm_state_probs,     
                    [current_decoded_state],      
                    [np.max(current_state_post)],
                    [norm_seq_pos]               
                ]
                feature_vector = np.concatenate(feature_vector_list)

                if len(feature_vector) != len(self.feature_names):
                    print(f"CRITICAL Error: Feature vector length ({len(feature_vector)}) "
                        f"does not match schema length ({len(self.feature_names)}).")
                    continue

                all_features.append(feature_vector)
                if for_training:
                    all_targets.append(seq[t + 1][self.target_sentiment_idx])
        
        X = np.array(all_features)
        y = np.array(all_targets) if for_training else None
        return X, y

    def _prepare_single_step_features_for_prediction(self, obs_prefix_sequence):
        """
        Helper to create the raw feature vector for predicting the step AFTER the last obs in prefix.
        The output of this method is NOT scaled or PCA-transformed.
        """
        if not self.is_hmm_trained: return None
        if obs_prefix_sequence.shape[0] == 0: return None
        if not self.feature_names: self._define_feature_schema()

        t = obs_prefix_sequence.shape[0] - 1
        
        try:
            state_posteriors_full = self.hmm_model.predict_proba(obs_prefix_sequence)
            decoded_states_full = self.hmm_model.predict(obs_prefix_sequence)
        except: return None

        current_state_post = state_posteriors_full[t]
        current_decoded_state = decoded_states_full[t]
        next_hmm_state_probs = current_state_post @ self.hmm_model.transmat_
        norm_seq_pos = t / (len(obs_prefix_sequence) -1) if len(obs_prefix_sequence) > 1 else 0.0
        
      
        feature_vector_list = [
            current_state_post,          
            next_hmm_state_probs,        
            [current_decoded_state],     
            [np.max(current_state_post)], 
            [norm_seq_pos]               
        ]
        feature_vector = np.concatenate(feature_vector_list)

        if len(feature_vector) != len(self.feature_names):
            print(f"Error in _prepare_single_step_features: Length mismatch. Got {len(feature_vector)}, expected {len(self.feature_names)}")
            return None
        return feature_vector.reshape(1, -1)

    def train_full_pipeline(self, all_observation_sequences, validation_split_ratio=0.2,
                            regression_train_split_ratio=0.8):
        print("=== Training Full HMM Regression Pipeline ===")
        
        if len(all_observation_sequences) < 10: 
            raise ValueError("Not enough observation sequences for robust train/validation split.")

        hmm_train_sequences, reg_and_val_sequences = train_test_split(
            all_observation_sequences, test_size=validation_split_ratio, 
            random_state=self.random_state, shuffle=True)
        
        print("\nStep 1: Fitting HMM Component...")
        if not self.fit_hmm(hmm_train_sequences): 
            print("HMM training failed. Aborting pipeline.")
            return

        print("\nStep 2: Extracting Features for Regression Models (from non-HMM-training data)...")
        X_reg_full, y_reg_full = self._extract_regression_features(reg_and_val_sequences, for_training=True)

        if X_reg_full.shape[0] == 0:
            print("Error: No features extracted for regression. Aborting.")
            return
            
        X_reg_full_processed = X_reg_full
        if self.feature_scaler:
            print("Fitting and applying feature scaler...")
            X_reg_full_processed = self.feature_scaler.fit_transform(X_reg_full_processed)
        
        if self.pca_transformer:
            print("Fitting and applying PCA...")
            X_reg_full_processed = self.pca_transformer.fit_transform(X_reg_full_processed)
            print(f"  PCA transformed features from {X_reg_full.shape[1]} to {X_reg_full_processed.shape[1]} dimensions.")
            if hasattr(self.pca_transformer, 'explained_variance_ratio_'):
                 print(f"  Explained variance ratio by PCA: {np.sum(self.pca_transformer.explained_variance_ratio_):.3f}")

        if X_reg_full_processed.shape[0] < 10: 
             print("Warning: Very few samples for regression training/validation after feature extraction.")
             X_reg_train, X_reg_val = X_reg_full_processed, np.array([])
             y_reg_train, y_reg_val = y_reg_full, np.array([])
        else:
            X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
                X_reg_full_processed, y_reg_full,
                test_size=(1-regression_train_split_ratio), 
                random_state=self.random_state + 10, shuffle=True 
            )
        print(f"Regression feature split: {X_reg_train.shape[0]} for training, {X_reg_val.shape[0]} for validation.")

        print("\nStep 4: Fitting Regression Components...")
        self.fit_regression(X_reg_train, y_reg_train, X_reg_val, y_reg_val) 

        if self.is_regression_trained:
            print("\nStep 5: Analyzing Feature Importance...")
            self.analyze_feature_importance()
        else:
            print("\nStep 5: Skipping Feature Importance (Regression not trained).")
        
        print("\n=== Pipeline Training Complete ===")

    def decode_hmm_sequence(self, observation_sequence):
        """Decodes the most likely HMM hidden state sequence."""
        if not self.is_hmm_trained:
            raise ValueError("HMM component is not trained yet.")
        if observation_sequence.ndim == 1:
            observation_sequence = observation_sequence.reshape(1, -1)
        if observation_sequence.shape[0] == 0:
            return np.array([], dtype=int)
        _log_prob, state_sequence = self.hmm_model.decode(observation_sequence, algorithm="viterbi")
        return state_sequence
    
    def analyze_hmm_states(self, observation_sequences, decoded_state_sequences, target_class_idx=TARGET_SENTIMENT):
        """
        Analyzes HMM states based on observations and decoded states,
        using probability thresholds for naming (like old HMMSurrogate).
        """
        if not self.is_hmm_trained:
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
        print("\n--- HMM State Analysis (Threshold-Based Naming) ---")
        for state, data in state_analysis.items():
            avg_prob = (data['sum_target_prob'] / data['occurrences']) if data['occurrences'] > 0 else 0.0
            results[state] = {
                'avg_prob_target_class': avg_prob,
                'num_occurrences': data['occurrences']
            }
            print(f"State {state}: Occurrences = {data['occurrences']}, Avg. P(Class {target_class_idx}) = {avg_prob:.3f}")
        
        state_names = {}
        for state_idx_loop in range(self.n_states):
            data_dict = results.get(state_idx_loop, {'avg_prob_target_class': 0.0, 'num_occurrences': 0})
            avg_prob = data_dict['avg_prob_target_class']
          
            if avg_prob <= PROB_THRESHOLDS["NEG_OBS"]:
                state_names[state_idx_loop] = "Leaning Negative/Low Confidence"
            elif avg_prob <= PROB_THRESHOLDS["NEU_OBS"]:
                state_names[state_idx_loop] = "Neutral/Uncertain"
            else:
                state_names[state_idx_loop] = "Leaning Positive/High Confidence"
      
        print(f"\nSuggested HMM State Interpretations (based on P(Class {target_class_idx})):")
        print_info = []
        for state_idx_loop in range(self.n_states):
            data_dict = results.get(state_idx_loop, {'avg_prob_target_class': 0.0, 'num_occurrences': 0})
            avg_prob_val = data_dict['avg_prob_target_class']
            name = state_names.get(state_idx_loop, f"State {state_idx_loop} (Unseen/Default)")
            print_info.append((state_idx_loop, avg_prob_val, name))
          
        print_info.sort(key=lambda x: x[1])
        for state_idx_sorted, avg_prob_sorted, name_sorted in print_info:
            print(f"  HMM State {state_idx_sorted}: ~{name_sorted} (Avg. P(Class {target_class_idx}) = {avg_prob_sorted:.3f})")
      
        results['state_names'] = state_names
        return results
    
    def _get_final_true_labels(self, observation_sequences):
        """Extract binary final sentiment labels from observation sequences."""
        final_labels = []
        for seq in observation_sequences:
            if seq.shape[0] > 0:
               
                final_prob_positive = seq[-1, self.target_sentiment_idx]
                
                final_labels.append(1 if final_prob_positive >= 0.5 else 0)
            else:
                final_labels.append(None) 
        return final_labels

    def _extract_final_prediction_features(self, observation_sequences, k_prefix=5):
        """Extract HMM features from sequence prefixes for final sentiment prediction."""
        if not self.is_hmm_trained:
            raise ValueError("HMM must be trained first.")
        
        all_prefix_features = []
        
        for seq in tqdm(observation_sequences, desc="Extracting final prediction features from prefixes"):
            if seq.shape[0] >= k_prefix:
                prefix_obs = seq[:k_prefix]
                try:
                    # Get HMM state posterior after k_prefix steps
                    # predict_proba returns P(S_t | o_1...o_t) for each t in the prefix
                    # We want the one for the last step of the prefix (k_prefix-1 index)
                    hmm_state_posteriors_at_k = self.hmm_model.predict_proba(prefix_obs)[-1]  # Shape: (n_states,)
                    
                    # We could add more HMM-derived features here if desired
                    #decoded_state_at_k = self.hmm_model.predict(prefix_obs)[-1]
                    #feature_vector = np.concatenate([hmm_state_posteriors_at_k, [decoded_state_at_k]])
                    feature_vector = hmm_state_posteriors_at_k
                    
                  
                    all_prefix_features.append(feature_vector)
                except Exception as e:
                    print(f"Warning: Could not process prefix for a sequence: {e}")
                    all_prefix_features.append(None)  # Placeholder for sequences that fail
            else:
                all_prefix_features.append(None)  # Placeholder for sequences shorter than k_prefix
                
        return all_prefix_features  

    def train_final_sentiment_classifier(self, all_observation_sequences, k_prefix=5):
        """Train a classifier to predict final sentiment from sequence prefixes."""
        print(f"\nTraining final sentiment classifier based on {k_prefix}-token prefixes...")
        
    
        raw_features_list = self._extract_final_prediction_features(all_observation_sequences, k_prefix=k_prefix)
        
  
        true_final_labels_list = self._get_final_true_labels(all_observation_sequences)
        
        # 3. Filter out None entries (from short sequences or processing errors)
      
        X_final_clf = []
        y_final_clf = []
        for features, label in zip(raw_features_list, true_final_labels_list):
            if features is not None and label is not None:
                X_final_clf.append(features)
                y_final_clf.append(label)
                
        if not X_final_clf:
            print("Error: No valid features extracted for training the final sentiment classifier.")
            return False
            
        X_final_clf = np.array(X_final_clf)
        y_final_clf = np.array(y_final_clf)
        
        print(f"Extracted {X_final_clf.shape[0]} feature vectors for final classification.")
       
        X_train, X_val, y_train, y_val = train_test_split(
            X_final_clf, y_final_clf, 
            test_size=0.2, 
            random_state=self.random_state + 300, 
            stratify=y_final_clf if np.unique(y_final_clf).size > 1 else None
        )

   
        if self.use_final_pred_scaling:
            X_train_processed = self.final_pred_feature_scaler.fit_transform(X_train)
            X_val_processed = self.final_pred_feature_scaler.transform(X_val)
        else:
            X_train_processed = X_train 
            X_val_processed = X_val

     
        print("Fitting final sentiment classifier...")
        self.final_sentiment_classifier.fit(X_train_processed, y_train)
        self.is_final_classifier_trained = True
        
   
        if X_val_processed.shape[0] > 0:
            val_accuracy = self.final_sentiment_classifier.score(X_val_processed, y_val)
            print(f"Validation Accuracy of final sentiment classifier: {val_accuracy:.4f}")
        
        print("Final sentiment classifier training complete.")
        return True

    def predict_final_sentiment_from_prefix(self, obs_prefix_k_steps):
        """Predict final sentiment from a sequence prefix."""
        if not self.is_hmm_trained:
            raise ValueError("HMM component is not trained.")
        if not self.is_final_classifier_trained:
            raise ValueError("Final sentiment classifier is not trained.")
            
        try:
            hmm_state_posteriors_at_k = self.hmm_model.predict_proba(obs_prefix_k_steps)[-1]
            feature_vector = hmm_state_posteriors_at_k.reshape(1, -1)  # Reshape for single sample
        except Exception as e:
            print(f"Error processing prefix for final prediction: {e}")
            return None, None
            
        if self.use_final_pred_scaling and hasattr(self.final_pred_feature_scaler, 'mean_'):
            feature_vector_processed = self.final_pred_feature_scaler.transform(feature_vector)
        else:
            feature_vector_processed = feature_vector

        final_prediction_label = self.final_sentiment_classifier.predict(feature_vector_processed)
        final_prediction_proba = self.final_sentiment_classifier.predict_proba(feature_vector_processed)
        
        return final_prediction_label[0], final_prediction_proba[0]
    

    def save_pipeline(self, path_prefix="models/hmm_regression_pipeline"):
        model_dir = os.path.dirname(path_prefix)
        if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir)
        
        pipeline_data = {
            'hmm_model': self.hmm_model if self.is_hmm_trained else None,
            'prob_regressor': None, 
            'lower_quantile_regressor': self.lower_quantile_regressor if hasattr(self, 'lower_quantile_regressor') else None,
            'upper_quantile_regressor': self.upper_quantile_regressor if hasattr(self, 'upper_quantile_regressor') else None,
            'prob_regressors_ensemble': None, 
            'feature_scaler': self.feature_scaler if self.use_feature_scaling and hasattr(self.feature_scaler, 'mean_') else None,
            'pca_transformer': self.pca_transformer if self.use_pca_features and hasattr(self.pca_transformer, 'mean_') else None,
            'final_sentiment_classifier': self.final_sentiment_classifier if self.is_final_classifier_trained else None,
            'final_pred_feature_scaler': self.final_pred_feature_scaler if self.use_final_pred_scaling and hasattr(self.final_pred_feature_scaler, 'mean_') else None,
            'config': {
                'n_states': self.n_states,
                'covariance_type': self.covariance_type,
                'regression_method': self.regression_method,
                'is_hmm_trained': self.is_hmm_trained,
                'is_regression_trained': self.is_regression_trained,
                'is_final_classifier_trained': self.is_final_classifier_trained,
                'feature_names': self.feature_names,
                'use_feature_scaling': self.use_feature_scaling,
                'use_pca_features': self.use_pca_features,
                'n_pca_components': self.n_pca_components,
                'use_final_pred_scaling': self.use_final_pred_scaling,
                'target_sentiment_idx': self.target_sentiment_idx
            }
        }
        if self.regression_method == 'ensemble':
            pipeline_data['prob_regressors_ensemble'] = self.prob_regressors if self.is_regression_trained else None
        else:
            pipeline_data['prob_regressor'] = self.prob_regressor if self.is_regression_trained else None

        joblib.dump(pipeline_data, f"{path_prefix}.pkl")
        print(f"Full HMM Regression Pipeline saved to {path_prefix}.pkl")

    @classmethod
    def load_pipeline(cls, path_prefix="models/hmm_regression_pipeline"):
        pipeline_data = joblib.load(f"{path_prefix}.pkl")
        config = pipeline_data['config']
        
        instance = cls(
            n_states=config['n_states'],
            covariance_type=config['covariance_type'],
            regression_method=config['regression_method'],
            use_feature_scaling=config.get('use_feature_scaling', True), 
            use_pca_features=config.get('use_pca_features', False),
            n_pca_components=config.get('n_pca_components', 10)
        )
        
        if pipeline_data['hmm_model']:
            instance.hmm_model = pipeline_data['hmm_model']
            instance.is_hmm_trained = config['is_hmm_trained']
        
        if config['is_regression_trained']:
            if instance.regression_method == 'ensemble':
                instance.prob_regressors = pipeline_data['prob_regressors_ensemble']
            else:
                instance.prob_regressor = pipeline_data['prob_regressor']
            instance.lower_quantile_regressor = pipeline_data.get('lower_quantile_regressor')
            instance.upper_quantile_regressor = pipeline_data.get('upper_quantile_regressor')
            instance.is_regression_trained = True
            
        if config.get('is_final_classifier_trained', False):
            instance.final_sentiment_classifier = pipeline_data['final_sentiment_classifier']
            instance.is_final_classifier_trained = True
            
        if pipeline_data.get('feature_scaler'):
            instance.feature_scaler = pipeline_data['feature_scaler']
        if pipeline_data.get('pca_transformer'):
            instance.pca_transformer = pipeline_data['pca_transformer']
        if pipeline_data.get('final_pred_feature_scaler'):
            instance.final_pred_feature_scaler = pipeline_data['final_pred_feature_scaler']
            
        instance.feature_names = config.get('feature_names', [])
        instance.target_sentiment_idx = config.get('target_sentiment_idx', TARGET_SENTIMENT)
        instance.use_final_pred_scaling = config.get('use_final_pred_scaling', True)
        
        print(f"Full HMM Regression Pipeline loaded from {path_prefix}.pkl")
        return instance
    
    def analyze_feature_importance(self):
        """Analyze feature importance for the probability regressor."""
        if not self.is_regression_trained:
            print("Regression models not trained. Cannot analyze.")
            return None
        if not self.feature_names:
            if self.is_hmm_trained and hasattr(self.hmm_model, 'means_'):
                self._define_feature_schema()
            if not self.feature_names:
                print("Feature names not defined. Cannot analyze importance without schema.")
                return None

        current_feature_names_for_regressor = self.feature_names
        
        if self.use_pca_features and self.pca_transformer and hasattr(self.pca_transformer, 'n_components_'):
            num_pca_components = self.pca_transformer.n_components_
            current_feature_names_for_regressor = [f'pca_comp_{i}' for i in range(num_pca_components)]
            print(f"Note: Feature importance is shown for {num_pca_components} PCA components.")
        elif self.use_feature_scaling and self.feature_scaler and hasattr(self.feature_scaler, 'n_features_in_'):
            if len(self.feature_names) != self.feature_scaler.n_features_in_:
                print(f"Warning: Mismatch between defined feature names ({len(self.feature_names)}) and scaler's input features ({self.feature_scaler.n_features_in_}). Using defined names.")

        importance_results = {}
        regressor_for_importance = None
        importance_attr_name = 'feature_importances_' 

        if self.regression_method == 'random_forest': 
            regressor_for_importance = self.prob_regressor
        elif self.regression_method == 'ensemble':
            rf_models = [reg for reg in self.prob_regressors if isinstance(reg, (RandomForestRegressor, GradientBoostingRegressor)) and hasattr(reg, importance_attr_name)]
            if rf_models:
                importances_list = [getattr(rf, importance_attr_name) for rf in rf_models]
                if all(len(imp) == len(current_feature_names_for_regressor) for imp in importances_list):
                    avg_importance = np.mean(importances_list, axis=0)
                    importance_results = {'feature_names': current_feature_names_for_regressor, 'avg_ensemble_prob_importance': avg_importance}
                    print("\n--- Ensemble Probability Regressor Feature Importance ---")
                    indices = np.argsort(avg_importance)[::-1][:min(10, len(current_feature_names_for_regressor))]
                    for i, idx in enumerate(indices): print(f"  {i+1}. {current_feature_names_for_regressor[idx]}: {avg_importance[idx]:.4f}")
                else: 
                    print(f"Warning: Feature name/importance length mismatch for ensemble. Expected {len(current_feature_names_for_regressor)} features for regressor.")
                    for i, imp in enumerate(importances_list):
                        print(f"  Model {i} importance length: {len(imp)}")
            else: print("No RandomForest/GradientBoosting models with feature importance in ensemble.")
        elif self.regression_method == 'mlp':
            print("Feature importance for MLP often requires permutation_importance (more intensive).")

        if regressor_for_importance and hasattr(regressor_for_importance, importance_attr_name):
            prob_importance = getattr(regressor_for_importance, importance_attr_name)
            if len(prob_importance) == len(current_feature_names_for_regressor):
                importance_results['probability_importance'] = prob_importance
                print("\n--- Probability Regressor Feature Importance ---")
                prob_indices = np.argsort(prob_importance)[::-1][:min(10, len(current_feature_names_for_regressor))]
                for i, idx in enumerate(prob_indices): print(f"  {i+1}. {current_feature_names_for_regressor[idx]}: {prob_importance[idx]:.4f}")
            else: 
                print(f"Warning: Prob regressor feature name/importance length mismatch. Expected {len(current_feature_names_for_regressor)}, got {len(prob_importance)}.")
        
        if self.regression_method == 'random_forest': 
            if self.lower_quantile_regressor and hasattr(self.lower_quantile_regressor, importance_attr_name):
                lower_q_imp = getattr(self.lower_quantile_regressor, importance_attr_name)
                if len(lower_q_imp) == len(current_feature_names_for_regressor):
                    importance_results['lower_quantile_importance'] = lower_q_imp
                    print("\n--- Lower Quantile Regressor Feature Importance ---")
                    lq_indices = np.argsort(lower_q_imp)[::-1][:min(5, len(current_feature_names_for_regressor))] 
                    for i, idx in enumerate(lq_indices): print(f"  {i+1}. {current_feature_names_for_regressor[idx]}: {lower_q_imp[idx]:.4f}")
                else: print(f"Warning: Lower quantile regressor feature name/importance length mismatch.")
            
            if self.upper_quantile_regressor and hasattr(self.upper_quantile_regressor, importance_attr_name):
                upper_q_imp = getattr(self.upper_quantile_regressor, importance_attr_name)
                if len(upper_q_imp) == len(current_feature_names_for_regressor):
                    importance_results['upper_quantile_importance'] = upper_q_imp
                    print("\n--- Upper Quantile Regressor Feature Importance ---")
                    uq_indices = np.argsort(upper_q_imp)[::-1][:min(5, len(current_feature_names_for_regressor))] 
                    for i, idx in enumerate(uq_indices): print(f"  {i+1}. {current_feature_names_for_regressor[idx]}: {upper_q_imp[idx]:.4f}")
                else: print(f"Warning: Upper quantile regressor feature name/importance length mismatch.")

        return importance_results
    
  