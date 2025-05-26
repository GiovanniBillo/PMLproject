import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np

from config import MODEL_NAME, DEVICE, MAX_TOKENS


class BlackBoxSentimentClassifier:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, max_tokens=MAX_TOKENS):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"Model loaded successfully on {self.device}")
        print(f"loaded black box model: {self.model_name}")
        print(f"Labels: {self.id2label}") 

    @torch.no_grad()
    def predict_proba_for_prefix(self, input_ids_prefix):
        
        if not isinstance(input_ids_prefix, torch.Tensor):
            input_ids_prefix = torch.tensor(input_ids_prefix).to(self.device)
        
        # Ensure input_ids_prefix is at least 1D before unsqueeze
        if input_ids_prefix.ndim == 0: # handle single token id
            input_ids_prefix = input_ids_prefix.unsqueeze(0)
            
        input_ids_prefix = input_ids_prefix.unsqueeze(0).to(self.device) # Add batch dimension
        outputs = self.model(input_ids_prefix)
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities.squeeze(0).cpu().numpy()
    
    def get_incremental_predictions(self, input_ids_full, max_len=None):
        if max_len is None:
            max_len = self.max_tokens
            
        prob_trajectory=[]
        if self.tokenizer.cls_token_id is not None and (not input_ids_full or input_ids_full[0] != self.tokenizer.cls_token_id):
            input_ids_core = [self.tokenizer.cls_token_id]+input_ids_full
        else:
            input_ids_core = list(input_ids_full) # Ensure it's a list for slicing and modification
        
        actual_len = min(max_len, len(input_ids_core))
        
        # Ensure we don't try to predict on an empty prefix if cls_token_id was the only token and actual_len is 0
        # or if input_ids_core is empty.
        # The loop range starts from 1, so it handles empty input_ids_core correctly (loop won't run).
        # Smallest prefix should be at least [CLS] token if present.
        start_index = 1 
        
        for i in range(start_index, actual_len + 1):
            current_prefix_ids = input_ids_core[:i]
            if not current_prefix_ids: # Should not happen with start_index=1 unless input_ids_core is empty
                continue
            probs = self.predict_proba_for_prefix(current_prefix_ids)
            prob_trajectory.append(probs)
        
        if not prob_trajectory: # Handle cases where no predictions were made (e.g. empty input_ids_full)
            return np.array([]).reshape(0, len(self.id2label)) # Return empty array with correct second dimension

        return np.array(prob_trajectory)
        
      
def log_inference_trajectories(processed_data, black_box_model, max_len=MAX_TOKENS):
    all_trajectories = []
    
    for item in tqdm(processed_data, desc="Processing data"):
        input_ids = item['input_ids']
        trajectory = black_box_model.get_incremental_predictions(input_ids, max_len=max_len)
        
        if trajectory.shape[0]>0:
            all_trajectories.append(trajectory)
    return all_trajectories


if __name__ == '__main__':
    from data_utils import get_tokenizer, load_imdb_data, preprocess_data_for_inference_logging
 
    bb_model = BlackBoxSentimentClassifier()
    tokenizer = bb_model.tokenizer 

   
    sample_dataset = load_imdb_data(split='test', num_samples=3, shuffle=False)
    processed_samples = preprocess_data_for_inference_logging(sample_dataset, tokenizer)

    
    trajectories = log_inference_trajectories(processed_samples, bb_model, max_len=20)

    for i, traj in enumerate(trajectories):
        print(f"\nTrajectory for sample {i+1} (Text: '{processed_samples[i]['text'][:50]}...'):")
        print(f"Shape: {traj.shape}")
        print(f"First 3 steps (Probabilities for {bb_model.id2label}):\n{traj[:3]}")
        assert traj.shape[1] == len(bb_model.id2label)