from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import numpy as np

from config import DATASET_NAME, MAX_TOKENS, MODEL_NAME

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_imdb_data(split ='train', num_samples=None,shuffle=True):

    dataset = load_dataset(DATASET_NAME, split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    if shuffle:
        dataset = dataset.shuffle(seed = 123)

    return dataset

def preprocess_data_for_inference_logging(dataset, tokenizer):
    processed_data = []
    
    for example in tqdm(dataset):
        text = example["text"]
        
        tokenized_output = tokenizer(text, truncation= False, padding=False)
        
        input_ids = tokenized_output["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        processed_data.append({"text": text,"input_ids": input_ids, "tokens": tokens})
        
    return processed_data

if __name__ == '__main__':
 
    tokenizer = get_tokenizer()
    train_dataset = load_imdb_data(split='train', num_samples=2)
    
    processed_train_data = preprocess_data_for_inference_logging(train_dataset, tokenizer)
    
    for item in processed_train_data:
        print(f"\nText: {item['text'][:100]}...")
        print(f"First 10 Tokens: {item['tokens'][:10]}")
        print(f"First 10 Input IDs: {item['input_ids'][:10]}")
        assert len(item['tokens']) == len(item['input_ids'])
    print(f"\nProcessed {len(processed_train_data)} samples.")