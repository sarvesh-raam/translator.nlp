"""TensorFlow-free inference: the encoder, Bahdanau attention, decoder and beam search from model.py,
re-implemented in NumPy so the server fits in a small (512 MB) free hosting instance.
Reads the files written by export_numpy.py."""
import os
import json
import numpy as np
from config import MODELS_DIR, MAX_SEQ_LEN

NEG_INF = -1e9

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

def gru_step(x, h, kernel, recurrent_kernel, bias):
    """One step of Keras' GRU (reset_after=True, the default): gates are ordered z, r, h and
    bias[0]/bias[1] are the input/recurrent biases."""
    units = h.shape[-1]
    xm = x @ kernel + bias[0]
    hm = h @ recurrent_kernel + bias[1]
    z = sigmoid(xm[:, :units] + hm[:, :units])
    r = sigmoid(xm[:, units:2 * units] + hm[:, units:2 * units])
    candidate = np.tanh(xm[:, 2 * units:] + r * hm[:, 2 * units:])
    return z * h + (1.0 - z) * candidate

class Translator:
    def __init__(self, models_dir=MODELS_DIR):
        with np.load(os.path.join(models_dir, 'nmt_weights.npz')) as weights:
            self.w = {name: weights[name] for name in weights.files}
        with open(os.path.join(models_dir, 'vocab.json'), encoding='utf-8') as f:
            vocab = json.load(f)
        self.en_word_index = vocab['en_word_index']
        self.en_oov_id = vocab['en_oov_id']
        self.hi_index_word = {int(i): word for i, word in vocab['hi_index_word'].items()}
        self.start_id = vocab['start_id']
        self.end_id = vocab['end_id']

    def encode_text(self, sentence):
        """texts_to_sequences + pad_sequences(truncating='post') for one already-cleaned sentence."""
        return [self.en_word_index.get(word, self.en_oov_id) for word in sentence.split()][:MAX_SEQ_LEN]

    def encode(self, ids):
        """Encoder outputs for the real (non-padding) tokens and the final state. Padding comes after
        the sentence and the masked GRU skips it, so only the real tokens need to be run."""
        w = self.w
        x = w['enc_embedding'][ids]
        h = np.zeros((1, w['enc_recurrent_kernel'].shape[0]), dtype=np.float32)
        outputs = []
        for t in range(len(ids)):
            h = gru_step(x[t:t + 1], h, w['enc_kernel'], w['enc_recurrent_kernel'], w['enc_bias'])
            outputs.append(h[0])
        return np.stack(outputs), h

    def decode_step(self, tokens, h, enc_out, enc_keys):
        """Decoder step for every beam at once: tokens (K,), h (K, units).
        Padding positions are left out of enc_out, which matches masking them with -1e9."""
        w = self.w
        query = h @ w['att_W1_kernel'] + w['att_W1_bias']
        score = np.tanh(query[:, np.newaxis, :] + enc_keys) @ w['att_V_kernel'] + w['att_V_bias']
        score = score[:, :, 0]
        attention = np.exp(score - score.max(axis=1, keepdims=True))
        attention /= attention.sum(axis=1, keepdims=True)
        context = attention @ enc_out

        x = np.concatenate([context, w['dec_embedding'][tokens]], axis=-1)
        # model.Decoder calls its GRU without initial_state, so each step starts from zeros;
        # the previous state only reaches the decoder through the attention query
        h = gru_step(x, np.zeros_like(h), w['dec_kernel'], w['dec_recurrent_kernel'], w['dec_bias'])
        logits = h @ w['fc_kernel'] + w['fc_bias']
        return logits, h, attention

    def beam_search(self, ids, beam_size=5, alpha=0.6, max_len=MAX_SEQ_LEN):
        """Same algorithm as model.beam_search_decode for one sentence: returns the decoded ids and the
        attention over the source tokens, shaped (steps, len(ids))."""
        K = beam_size
        enc_out, h = self.encode(ids)
        enc_keys = enc_out @ self.w['att_W2_kernel'] + self.w['att_W2_bias']
        vocab = self.w['fc_bias'].shape[0]

        h = np.repeat(h, K, axis=0)
        # Only the first beam is live at the start, so the K beams don't all pick the same token
        scores = np.array([0.0] + [NEG_INF] * (K - 1), dtype=np.float32)
        lengths = np.zeros(K, dtype=np.float32)
        finished = np.zeros(K, dtype=bool)
        prev_id = np.full(K, -1)
        tokens = np.zeros((K, 0), dtype=np.int64)
        attentions = np.zeros((K, 0, len(ids)), dtype=np.float32)
        # A finished beam only "emits" padding, at no cost, so its score stays fixed
        finished_row = np.full(vocab, NEG_INF, dtype=np.float32)
        finished_row[0] = 0.0

        dec_input = np.full(K, self.start_id)
        for _ in range(max_len):
            logits, h, attention = self.decode_step(dec_input, h, enc_out, enc_keys)
            log_probs = log_softmax(logits)
            log_probs[:, 0] += NEG_INF  # padding is never emitted
            live = prev_id >= 0
            log_probs[np.arange(K)[live], prev_id[live]] += NEG_INF  # no immediate repeats
            log_probs[finished] = finished_row

            candidates = (scores[:, np.newaxis] + log_probs).reshape(-1)
            flat = np.argsort(-candidates, kind='stable')[:K]
            scores = candidates[flat]
            parent, next_id = flat // vocab, flat % vocab

            # Reorder every per-beam state to follow the beam each candidate came from
            parent_finished = finished[parent]
            lengths = lengths[parent] + (~parent_finished)
            finished = parent_finished | (next_id == self.end_id)
            tokens = np.concatenate([tokens[parent], next_id[:, np.newaxis]], axis=1)
            attentions = np.concatenate([attentions[parent], attention[parent][:, np.newaxis, :]], axis=1)
            h = h[parent]

            if finished.all():
                break
            prev_id = next_id
            dec_input = next_id

        best = int(np.argmax(scores / np.maximum(lengths, 1.0) ** alpha))
        return tokens[best], attentions[best]

    def translate(self, sentence):
        """Translate an already-cleaned English sentence. Returns (hindi, attention) where attention
        is (output words, source words)."""
        ids = self.encode_text(sentence)
        if not ids:
            return "", np.zeros((0, 0))
        out_ids, attention = self.beam_search(ids)
        words = []
        for idx in out_ids:
            idx = int(idx)
            if idx in (0, self.end_id):
                break
            words.append(self.hi_index_word.get(idx, '<unk>'))
        return ' '.join(words), attention
