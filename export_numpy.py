"""Export the trained TensorFlow model to plain NumPy/JSON files for the TensorFlow-free server.
Run once after every training run:  python export_numpy.py"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import tensorflow as tf
from config import DATA_DIR, MODELS_DIR, MAX_SEQ_LEN, EMBEDDING_DIM, UNITS
from model import Encoder, Decoder
from preprocess import vocab_size

def export():
    with open(os.path.join(DATA_DIR, 'en_tokenizer.pkl'), 'rb') as f:
        en_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'hi_tokenizer.pkl'), 'rb') as f:
        hi_tokenizer = pickle.load(f)
    start_id = hi_tokenizer.word_index['<start>']

    encoder = Encoder(vocab_size(en_tokenizer), EMBEDDING_DIM, UNITS, 1)
    decoder = Decoder(vocab_size(hi_tokenizer), EMBEDDING_DIM, UNITS, 1)
    # Build the models before loading (non-zero ids: cuDNN's GRU rejects an all-padding sequence)
    enc_out, enc_hidden, enc_mask = encoder(tf.ones((1, MAX_SEQ_LEN), dtype=tf.int32), encoder.initialize_hidden_state())
    decoder(tf.constant([[start_id]]), enc_hidden, enc_out, enc_mask)
    encoder.load_weights(os.path.join(MODELS_DIR, 'encoder_weights.weights.h5'))
    decoder.load_weights(os.path.join(MODELS_DIR, 'decoder_weights.weights.h5'))

    attention = decoder.attention
    weights = {
        'enc_embedding': encoder.embedding.embeddings,
        'enc_kernel': encoder.gru.cell.kernel,
        'enc_recurrent_kernel': encoder.gru.cell.recurrent_kernel,
        'enc_bias': encoder.gru.cell.bias,
        'dec_embedding': decoder.embedding.embeddings,
        'dec_kernel': decoder.gru.cell.kernel,
        'dec_recurrent_kernel': decoder.gru.cell.recurrent_kernel,
        'dec_bias': decoder.gru.cell.bias,
        'att_W1_kernel': attention.W1.kernel, 'att_W1_bias': attention.W1.bias,
        'att_W2_kernel': attention.W2.kernel, 'att_W2_bias': attention.W2.bias,
        'att_V_kernel': attention.V.kernel, 'att_V_bias': attention.V.bias,
        'fc_kernel': decoder.fc.kernel, 'fc_bias': decoder.fc.bias,
    }
    np.savez(os.path.join(MODELS_DIR, 'nmt_weights.npz'),
             **{name: np.asarray(w.numpy(), dtype=np.float32) for name, w in weights.items()})

    # Only ids the model can see (< num_words); anything else maps to <unk>, as in texts_to_sequences
    en_vocab = vocab_size(en_tokenizer)
    hi_vocab = vocab_size(hi_tokenizer)
    vocab = {
        'en_word_index': {w: i for w, i in en_tokenizer.word_index.items() if i < en_vocab},
        'en_oov_id': en_tokenizer.word_index[en_tokenizer.oov_token],
        'hi_index_word': {i: w for i, w in hi_tokenizer.index_word.items() if i < hi_vocab},
        'start_id': start_id,
        'end_id': hi_tokenizer.word_index['<end>'],
    }
    with open(os.path.join(MODELS_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Exported weights and vocab to {MODELS_DIR}/")

if __name__ == "__main__":
    export()
