import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warnings
import tensorflow as tf
import argparse
import glob
import json
import shutil
import time
import numpy as np
import pickle
import sacrebleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import DATA_DIR, MODELS_DIR, MAX_SEQ_LEN, EMBEDDING_DIM, UNITS
from model import Encoder, Decoder, greedy_decode, beam_search_decode, ids_to_words
from preprocess import vocab_size

# Parameters
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 3  # Stop after this many epochs without a better validation BLEU
EVAL_BATCH_SIZE = 64

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def load_data():
    en_input = np.load(os.path.join(DATA_DIR, 'en_input.npy'))
    hi_input = np.load(os.path.join(DATA_DIR, 'hi_input.npy'))

    with open(os.path.join(DATA_DIR, 'en_tokenizer.pkl'), 'rb') as f:
        en_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'hi_tokenizer.pkl'), 'rb') as f:
        hi_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'val_pairs.json'), encoding='utf-8') as f:
        val_pairs = json.load(f)
    with open(os.path.join(DATA_DIR, 'test_pairs.json'), encoding='utf-8') as f:
        test_pairs = json.load(f)

    return en_input, hi_input, en_tokenizer, hi_tokenizer, val_pairs, test_pairs

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
            enc_output, dec_hidden, enc_mask = encoder(inp, tf.zeros((batch, UNITS)), training=True)
            dec_input = tf.fill([batch, 1], start_id)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, enc_mask,
                                                     training=True)
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

def evaluate_bleu(test_en, test_hi, encoder, decoder, en_tokenizer, hi_tokenizer, decode=greedy_decode):
    """Corpus-level BLEU on an official IITB split (never trained on), via sacrebleu.
    Validation uses fast greedy decoding; the final test score uses beam search, like the app."""
    start_id = hi_tokenizer.word_index['<start>']
    end_id = hi_tokenizer.word_index['<end>']
    sources = pad_sequences(en_tokenizer.texts_to_sequences(test_en), maxlen=MAX_SEQ_LEN,
                            padding='post', truncating='post')

    hypotheses = []
    for i in range(0, len(sources), EVAL_BATCH_SIZE):
        ids, _ = decode(encoder, decoder, tf.constant(sources[i:i + EVAL_BATCH_SIZE]),
                               start_id, end_id, MAX_SEQ_LEN)
        hypotheses += [' '.join(ids_to_words(row, hi_tokenizer.index_word, end_id)) for row in ids]
    return sacrebleu.corpus_bleu(hypotheses, [list(test_hi)]).score

def save_checkpoint(manager, checkpoint_dir, attempts=5):
    """On Windows, antivirus/indexers can briefly lock the file TF renames while saving.
    Retry instead of losing the run, and keep training if it still fails."""
    for attempt in range(1, attempts + 1):
        # A failed save leaves a <prefix>_temp folder behind
        for leftover in glob.glob(os.path.join(checkpoint_dir, '*_temp')):
            shutil.rmtree(leftover, ignore_errors=True)
        try:
            return manager.save()
        except tf.errors.OpError as e:
            print(f"Checkpoint save failed (attempt {attempt}/{attempts}): {e.message.splitlines()[0]}")
            time.sleep(5 * attempt)
    print("WARNING: could not save a checkpoint this epoch; training continues.")
    return None

def run_training(epochs=EPOCHS, limit=None, bleu_limit=None, resume=False):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    checkpoint_dir = os.path.join(MODELS_DIR, 'checkpoints')
    progress_path = os.path.join(checkpoint_dir, 'progress.json')
    encoder_path = os.path.join(MODELS_DIR, 'encoder_weights.weights.h5')
    decoder_path = os.path.join(MODELS_DIR, 'decoder_weights.weights.h5')

    en_input, hi_input, en_tokenizer, hi_tokenizer, val_pairs, test_pairs = load_data()
    en_input, hi_input = en_input[:limit], hi_input[:limit]
    val_en, val_hi = val_pairs['en'][:bleu_limit], val_pairs['hi'][:bleu_limit]
    test_en, test_hi = test_pairs['en'][:bleu_limit], test_pairs['hi'][:bleu_limit]

    dataset = tf.data.Dataset.from_tensor_slices((en_input, hi_input)).shuffle(len(en_input))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    encoder = Encoder(vocab_size(en_tokenizer), EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_size(hi_tokenizer), EMBEDDING_DIM, UNITS, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    train_step = make_train_step(encoder, decoder, optimizer, hi_tokenizer.word_index['<start>'])

    # Saved after every epoch, so an interrupted run loses at most one epoch
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

    progress = {"loss": [], "val_bleu": []}
    if resume:
        if manager.latest_checkpoint is None:
            raise SystemExit(f"--resume given but no checkpoint found in {checkpoint_dir}")
        # Build the models and optimizer variables so the restore applies to them.
        # Non-zero ids: cuDNN's GRU rejects an all-padding (fully masked) sequence.
        train_step(tf.ones((1, MAX_SEQ_LEN), tf.int32), tf.ones((1, MAX_SEQ_LEN), tf.int32))
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        with open(progress_path) as f:
            progress = json.load(f)
        print(f"Resumed from {manager.latest_checkpoint} after epoch {len(progress['loss'])}")

    print("--- Neural Training Initiated ---")
    print(f"Dataset Size: {len(en_input)} pairs (+{len(val_en)} validation, {len(test_en)} test pairs)")
    print(f"Batch Size: {BATCH_SIZE} | Units: {UNITS}")
    print(f"Max Epochs: {epochs} (stops early after {PATIENCE} epochs without a better validation BLEU)")
    print("Please wait while the model compiles and shuffles data...")

    print(f"\nStarting training for up to {epochs} epochs...")

    for epoch in range(len(progress['loss']), epochs):
        val_bleus = progress['val_bleu']
        best_epoch = val_bleus.index(max(val_bleus)) + 1 if val_bleus else 0
        if val_bleus and len(val_bleus) - best_epoch >= PATIENCE:
            print(f"No validation improvement for {PATIENCE} epochs, stopping early.")
            break

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
        val_bleu = evaluate_bleu(val_en, val_hi, encoder, decoder, en_tokenizer, hi_tokenizer)
        print(f'Epoch {epoch+1} Loss {epoch_loss:.4f} | Validation BLEU {val_bleu:.2f}')

        # The served weights are always those of the best epoch so far, not the last one
        if val_bleu > max(val_bleus, default=-1.0):
            encoder.save_weights(encoder_path)
            decoder.save_weights(decoder_path)
            print('New best validation BLEU, weights saved.')

        # progress.json only advances with a saved checkpoint, so --resume stays consistent
        if save_checkpoint(manager, checkpoint_dir):
            progress = {"loss": progress['loss'] + [epoch_loss], "val_bleu": val_bleus + [val_bleu]}
            with open(progress_path, 'w') as f:
                json.dump(progress, f)
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec (checkpoint saved)\n')
        else:
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    # Score the best epoch, not the last one
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    best = max(progress['val_bleu'])
    print(f"\nBest validation BLEU {best:.2f} at epoch {progress['val_bleu'].index(best) + 1}")

    print(f"Evaluating BLEU on {len(test_en)} official test pairs...")
    bleu_score = evaluate_bleu(test_en, test_hi, encoder, decoder, en_tokenizer, hi_tokenizer,
                               decode=beam_search_decode)
    print(f"Test BLEU: {bleu_score:.2f}")

    # Save training metrics for the dashboard
    history = {
        "bleu": round(float(bleu_score), 2),
        "training_history": {
            "epochs": list(range(1, len(progress['loss']) + 1)),
            "loss": progress['loss'],
            "val_bleu": progress['val_bleu']
        }
    }
    with open(os.path.join(DATA_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print("Models and history saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="Train on only the first N pairs (smoke tests)")
    parser.add_argument("--bleu-limit", type=int, default=None, help="Score BLEU on only the first N validation/test pairs")
    parser.add_argument("--resume", action="store_true", help="Continue from the last saved epoch")
    parser.add_argument("--rescore", action="store_true",
                        help="Don't train: re-score the saved best weights on the test set and update history.pkl")
    args = parser.parse_args()
    if args.rescore:
        # Resuming with no epochs left skips straight to the final test evaluation
        run_training(0, args.limit, args.bleu_limit, resume=True)
    else:
        run_training(args.epochs, args.limit, args.bleu_limit, args.resume)
