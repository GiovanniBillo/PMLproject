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
            
            pass