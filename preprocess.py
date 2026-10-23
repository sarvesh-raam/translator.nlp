import os
import re
import json
import random
import numpy as np
import pickle
from config import DATA_DIR, MAX_SEQ_LEN

# Configuration
SENTENCE_LIMIT = 500000  # Training pairs sampled from the deduplicated corpus
MIN_SEQ_LEN = 3          # Drops the corpus' short software-UI strings ("armenian", "default window state")
VOCAB_SIZE = 30000       # Limit vocab size; rarer words become <unk>
SEED = 42

def clean_text(text, is_hindi=False):
    """
    Basic text cleaning pipeline, shared by preprocessing and inference.
    """
    text = text.lower().strip()
    # Pad sentence punctuation (incl. the Hindi danda) with spaces so it becomes its own token
    text = re.sub(r"([?.!,।॥])", r" \1 ", text)
    if not is_hindi:
        # English cleaning (keep digits and in-word apostrophes like "don't")
        text = re.sub(r"[^a-z0-9'?.!,]+", " ", text)
        text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    else:
        # Hindi cleaning (Allow Devanagari characters, digits and basic punctuation)
        text = re.sub(r"[^ऀ-ॿ0-9?.!,]+", " ", text)

    return " ".join(text.split())

def vocab_size(tokenizer):
    """Number of ids texts_to_sequences can produce (ids >= num_words are mapped to <unk>)."""
    return min(tokenizer.num_words, len(tokenizer.word_index) + 1)

def clean_pairs(rows):
    pairs = []
    for row in rows:
        en = clean_text(row['en'])
        hi = clean_text(row['hi'], is_hindi=True)
        # Hindi gets <start>/<end> added, so it may use two fewer words
        if MIN_SEQ_LEN <= len(en.split()) <= MAX_SEQ_LEN and 0 < len(hi.split()) <= MAX_SEQ_LEN - 2:
            pairs.append((en, hi))
    return pairs

def preprocess_and_save():
    # Only needed here, keeps clean_text importable by the TensorFlow-free server
    from datasets import load_dataset
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print("Loading the IIT Bombay English-Hindi corpus...")
    corpus = load_dataset("cfilt/iitb-english-hindi")

    # The corpus is heavily duplicated, so dedupe before sampling
    train_pairs = list(dict.fromkeys(clean_pairs(corpus['train']['translation'])))
    val_pairs = clean_pairs(corpus['validation']['translation'])
    test_pairs = clean_pairs(corpus['test']['translation'])
    print(f"{len(train_pairs)} unique training pairs, {len(test_pairs)} test pairs after cleaning.")

    # Never train on a sentence that is used for evaluation
    eval_sources = {en for en, _ in val_pairs + test_pairs}
    train_pairs = [p for p in train_pairs if p[0] not in eval_sources]

    random.Random(SEED).shuffle(train_pairs)
    train_pairs = train_pairs[:SENTENCE_LIMIT]
    print(f"Sampled {len(train_pairs)} training pairs.")

    en_sentences = [en for en, _ in train_pairs]
    hi_sentences = ['<start> ' + hi + ' <end>' for _, hi in train_pairs]

    # English Tokenizer
    en_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', oov_token='<unk>')
    en_tokenizer.fit_on_texts(en_sentences)
    en_sequences = en_tokenizer.texts_to_sequences(en_sentences)
    en_padded = pad_sequences(en_sequences, padding='post', maxlen=MAX_SEQ_LEN)

    # Hindi Tokenizer (Target)
    hi_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', oov_token='<unk>')
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

    # Raw cleaned text, so BLEU references keep words that are <unk> to the tokenizer.
    # Validation picks the best epoch during training; test is only scored at the end.
    for name, pairs in (('val_pairs.json', val_pairs), ('test_pairs.json', test_pairs)):
        with open(os.path.join(DATA_DIR, name), 'w', encoding='utf-8') as f:
            json.dump({"en": [en for en, _ in pairs], "hi": [hi for _, hi in pairs]},
                      f, ensure_ascii=False)

    print(f"Preprocessing complete! Data and tokenizers saved to the '{DATA_DIR}/' directory.")
    return en_tokenizer, hi_tokenizer

if __name__ == "__main__":
    preprocess_and_save()
