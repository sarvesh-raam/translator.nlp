import os
import re
import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
DATA_DIR = "data"
SENTENCE_LIMIT = 30000  # Subset for faster training
MAX_SEQ_LEN = 30       # Max length of sentences
VOCAB_SIZE = 15000     # Limit vocab size

def clean_text(text, is_hindi=False):
    """
    Basic text cleaning pipeline.
    """
    text = text.lower().strip()
    if not is_hindi:
        # English cleaning
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r'[" "]', " ", text)
        text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    else:
        # Hindi cleaning (Allow Devanagari characters)
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r'[" "]', " ", text)
        # Regex for Devanagari characters and basic punctuation
        text = re.sub(r"[^\u0900-\u097F?.!,]+", " ", text)
    
    return text.strip()

def preprocess_and_save():
    print(f"Loading {SENTENCE_LIMIT} English-Hindi sentence pairs from IIT Bombay corpus...")
    dataset = load_dataset("cfilt/iitb-english-hindi", split=f"train[:{SENTENCE_LIMIT}]")
    
    en_sentences = []
    hi_sentences = []
    
    for item in dataset:
        en = clean_text(item['translation']['en'])
        hi = clean_text(item['translation']['hi'], is_hindi=True)
        hi = '<start> ' + hi + ' <end>'
        
        if len(en.split()) <= MAX_SEQ_LEN and len(hi.split()) <= MAX_SEQ_LEN:
            en_sentences.append(en)
            hi_sentences.append(hi)
            
    print(f"Loaded {len(en_sentences)} valid sentence pairs.")
    
    # English Tokenizer
    en_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='')
    en_tokenizer.fit_on_texts(en_sentences)
    en_sequences = en_tokenizer.texts_to_sequences(en_sentences)
    en_padded = pad_sequences(en_sequences, padding='post', maxlen=MAX_SEQ_LEN)
    
    # Hindi Tokenizer (Target)
    hi_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='')
    hi_tokenizer.fit_on_texts(hi_sentences)
    hi_sequences = hi_tokenizer.texts_to_sequences(hi_sentences)
    hi_padded = pad_sequences(hi_sequences, padding='post', maxlen=MAX_SEQ_LEN)
    
    # Save Tokenizers and Data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    with open(os.path.join(DATA_DIR, 'en_tokenizer.pkl'), 'wb') as f:
        pickle.dump(en_tokenizer, f)
    with open(os.path.join(DATA_DIR, 'hi_tokenizer.pkl'), 'wb') as f:
        pickle.dump(hi_tokenizer, f)
        
    np.save(os.path.join(DATA_DIR, 'en_input.npy'), en_padded)
    np.save(os.path.join(DATA_DIR, 'hi_input.npy'), hi_padded)
    
    print("Preprocessing complete! Data and tokenizers saved to the 'data/' directory.")
    return en_tokenizer, hi_tokenizer

if __name__ == "__main__":
    preprocess_and_save()
