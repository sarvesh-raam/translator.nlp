import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warnings
import tensorflow as tf
import argparse
import json
import time
import numpy as np
import pickle
import sacrebleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import DATA_DIR, MODELS_DIR, MAX_SEQ_LEN, EMBEDDING_DIM, UNITS
from model import Encoder, Decoder, greedy_decode, ids_to_words
from preprocess import vocab_size

# Parameters
BATCH_SIZE = 32
EPOCHS = 10
EVAL_BATCH_SIZE = 64

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def load_data():
    en_input = np.load(os.path.join(DATA_DIR, 'en_input.npy'))
    hi_input = np.load(os.path.join(DATA_DIR, 'hi_input.npy'))

    with open(os.path.join(DATA_DIR, 'en_tokenizer.pkl'), 'rb') as f:
        en_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'hi_tokenizer.pkl'), 'rb') as f:
        hi_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_pairs.json'), encoding='utf-8') as f:
        test_pairs = json.load(f)

    return en_input, hi_input, en_tokenizer, hi_tokenizer, test_pairs

def masked_loss(real, pred):
    """Summed cross-entropy over non-padding tokens, and the number of those tokens."""
    mask = tf.cast(tf.math.not_equal(real, 0), tf.float32)
    loss_ = loss_object(real, pred) * mask
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

def make_train_step(encoder, decoder, optimizer, start_id):
    @tf.function
    def train_step(inp, targ):
        batch = tf.shape(inp)[0]
        total_loss = 0.0
        total_tokens = 0.0
        with tf.GradientTape() as tape:
            enc_output, dec_hidden, enc_mask = encoder(inp, tf.zeros((batch, UNITS)))
            dec_input = tf.fill([batch, 1], start_id)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, enc_mask)
                step_loss, step_tokens = masked_loss(targ[:, t], predictions)
                total_loss += step_loss
                total_tokens += step_tokens
                dec_input = tf.expand_dims(targ[:, t], 1)

            # Mean loss per real (non-padding) target token
            loss = total_loss / tf.maximum(total_tokens, 1.0)

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss
    return train_step

def evaluate_bleu(test_en, test_hi, encoder, decoder, en_tokenizer, hi_tokenizer):
    """Corpus-level BLEU on the official IITB test split (never trained on), via sacrebleu."""
    start_id = hi_tokenizer.word_index['<start>']
    end_id = hi_tokenizer.word_index['<end>']
    sources = pad_sequences(en_tokenizer.texts_to_sequences(test_en), maxlen=MAX_SEQ_LEN,
                            padding='post', truncating='post')

    hypotheses = []
    for i in range(0, len(sources), EVAL_BATCH_SIZE):
        ids, _ = greedy_decode(encoder, decoder, tf.constant(sources[i:i + EVAL_BATCH_SIZE]),
                               start_id, end_id, MAX_SEQ_LEN)
        hypotheses += [' '.join(ids_to_words(row, hi_tokenizer.index_word, end_id)) for row in ids]
    return sacrebleu.corpus_bleu(hypotheses, [list(test_hi)]).score

def run_training(epochs=EPOCHS, limit=None, bleu_limit=None):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    en_input, hi_input, en_tokenizer, hi_tokenizer, test_pairs = load_data()
    en_input, hi_input = en_input[:limit], hi_input[:limit]
    test_en, test_hi = test_pairs['en'][:bleu_limit], test_pairs['hi'][:bleu_limit]

    dataset = tf.data.Dataset.from_tensor_slices((en_input, hi_input)).shuffle(len(en_input))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    encoder = Encoder(vocab_size(en_tokenizer), EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_size(hi_tokenizer), EMBEDDING_DIM, UNITS, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    train_step = make_train_step(encoder, decoder, optimizer, hi_tokenizer.word_index['<start>'])

    epoch_losses = []
    print("--- Neural Training Initiated ---")
    print(f"Dataset Size: {len(en_input)} pairs (+{len(test_en)} official test pairs for BLEU eval)")
    print(f"Batch Size: {BATCH_SIZE} | Units: {UNITS}")
    print(f"Estimated Epochs: {epochs}")
    print("Please wait while the model compiles and shuffles data...")

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        num_batches = 0

        for (batch, (inp, targ)) in enumerate(dataset):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss
            num_batches += 1

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        epoch_loss = float(total_loss / num_batches)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1} Loss {epoch_loss:.4f}')

        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    # Save weights and history
    encoder.save_weights(os.path.join(MODELS_DIR, 'encoder_weights.weights.h5'))
    decoder.save_weights(os.path.join(MODELS_DIR, 'decoder_weights.weights.h5'))

    print(f"\nEvaluating BLEU on {len(test_en)} official test pairs...")
    bleu_score = evaluate_bleu(test_en, test_hi, encoder, decoder, en_tokenizer, hi_tokenizer)
    print(f"Test BLEU: {bleu_score:.2f}")

    # Save training metrics for the dashboard
    history = {
        "bleu": round(float(bleu_score), 2),
        "training_history": {
            "epochs": list(range(1, epochs + 1)),
            "loss": epoch_losses
        }
    }
    with open(os.path.join(DATA_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print("Models and history saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="Train on only the first N pairs (smoke tests)")
    parser.add_argument("--bleu-limit", type=int, default=None, help="Score BLEU on only the first N test pairs")
    args = parser.parse_args()
    run_training(args.epochs, args.limit, args.bleu_limit)
