import tensorflow as tf
from config import DROPOUT

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(DROPOUT)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden, training=False):
        # mask shape == (batch_size, max_length); False on padding (id 0)
        mask = tf.math.not_equal(x, 0)
        x = self.dropout(self.embedding(x), training=training)
        # With the mask the GRU skips padding, so state is taken at the last real token
        output, state = self.gru(x, initial_state=hidden, mask=mask)
        return output, state, mask

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, values_mask=None):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # Padding positions get ~zero attention weight
        if values_mask is not None:
            score += (1.0 - tf.cast(values_mask, score.dtype))[:, :, tf.newaxis] * -1e9

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.dropout = tf.keras.layers.Dropout(DROPOUT)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, enc_mask=None, training=False):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output, enc_mask)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.dropout(self.embedding(x), training=training)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab_size)
        x = self.fc(self.dropout(output, training=training))

        return x, state, attention_weights

def greedy_decode(encoder, decoder, inputs, start_id, end_id, max_len):
    """Batched greedy decoding. Padding (id 0) is never emitted, and an immediate repeat of the
    previous token falls back to the runner-up (avoids '. . . .'-style degenerate loops).
    Returns numpy (ids, attention) shaped (batch, steps) and (batch, steps, source_len)."""
    batch = tf.shape(inputs)[0]
    enc_out, dec_hidden, enc_mask = encoder(inputs, tf.zeros((batch, encoder.enc_units)))

    dec_input = tf.fill([batch, 1], start_id)
    prev_id = tf.fill([batch], -1)
    finished = tf.zeros([batch], dtype=tf.bool)
    block_pad = tf.one_hot(0, decoder.fc.units, on_value=-1e9, off_value=0.0)

    ids, attentions = [], []
    for _ in range(max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, enc_mask)
        top2 = tf.math.top_k(predictions + block_pad, k=2).indices
        next_id = tf.where(tf.equal(top2[:, 0], prev_id), top2[:, 1], top2[:, 0])
        next_id = tf.where(finished, tf.zeros_like(next_id), next_id)

        ids.append(next_id)
        attentions.append(attention_weights[:, :, 0])
        finished = finished | tf.equal(next_id, end_id)
        if bool(tf.reduce_all(finished)):
            break

        prev_id = next_id
        dec_input = tf.expand_dims(next_id, 1)

    return tf.stack(ids, axis=1).numpy(), tf.stack(attentions, axis=1).numpy()

def beam_search_decode(encoder, decoder, inputs, start_id, end_id, max_len, beam_size=5, alpha=0.6):
    """Batched beam search: keeps the beam_size best partial translations per sentence and returns
    the one with the best length-normalised log-probability (score / length**alpha). Like
    greedy_decode, padding is never emitted and a token can't immediately repeat itself.
    Returns numpy (ids, attention) shaped (batch, steps) and (batch, steps, source_len)."""
    batch = tf.shape(inputs)[0]
    K = beam_size
    vocab = decoder.fc.units
    enc_out, dec_hidden, enc_mask = encoder(inputs, tf.zeros((batch, encoder.enc_units)))

    # Each sentence is decoded as K rows: (batch * K, ...)
    enc_out = tf.repeat(enc_out, K, axis=0)
    enc_mask = tf.repeat(enc_mask, K, axis=0)
    dec_hidden = tf.repeat(dec_hidden, K, axis=0)

    # Only the first beam is live at the start, so the K beams don't all pick the same token
    scores = tf.tile(tf.constant([[0.0] + [-1e9] * (K - 1)]), [batch, 1])
    lengths = tf.zeros([batch, K])
    finished = tf.zeros([batch, K], dtype=tf.bool)
    prev_id = tf.fill([batch, K], -1)
    tokens = tf.zeros([batch, K, 0], dtype=tf.int32)
    attentions = tf.zeros([batch, K, 0, tf.shape(inputs)[1]])

    block_pad = tf.one_hot(0, vocab, on_value=-1e9, off_value=0.0)
    # A finished beam only "emits" padding, at no cost, so its score stays fixed
    finished_row = tf.one_hot(0, vocab, on_value=0.0, off_value=-1e9)

    dec_input = tf.fill([batch * K, 1], start_id)
    for _ in range(max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, enc_mask)
        log_probs = tf.reshape(tf.nn.log_softmax(predictions), [batch, K, vocab])
        log_probs += block_pad + tf.one_hot(prev_id, vocab, on_value=-1e9, off_value=0.0)
        log_probs = tf.where(finished[:, :, tf.newaxis], finished_row, log_probs)

        candidates = tf.reshape(scores[:, :, tf.newaxis] + log_probs, [batch, K * vocab])
        scores, flat = tf.math.top_k(candidates, k=K)
        parent = flat // vocab
        next_id = flat % vocab

        # Reorder every per-beam state to follow the beam each candidate came from
        parent_finished = tf.gather(finished, parent, batch_dims=1)
        lengths = tf.gather(lengths, parent, batch_dims=1) + tf.cast(~parent_finished, tf.float32)
        finished = parent_finished | tf.equal(next_id, end_id)
        tokens = tf.concat([tf.gather(tokens, parent, batch_dims=1), next_id[:, :, tf.newaxis]], axis=2)
        step_attention = tf.gather(tf.reshape(attention_weights[:, :, 0], [batch, K, -1]), parent, batch_dims=1)
        attentions = tf.concat([tf.gather(attentions, parent, batch_dims=1),
                                step_attention[:, :, tf.newaxis, :]], axis=2)
        dec_hidden = tf.reshape(tf.gather(tf.reshape(dec_hidden, [batch, K, -1]), parent, batch_dims=1),
                                [batch * K, -1])

        if bool(tf.reduce_all(finished)):
            break
        prev_id = next_id
        dec_input = tf.reshape(next_id, [batch * K, 1])

    best = tf.argmax(scores / tf.maximum(lengths, 1.0) ** alpha, axis=1, output_type=tf.int32)
    ids = tf.gather(tokens, best, batch_dims=1)
    attention = tf.gather(attentions, best, batch_dims=1)
    return ids.numpy(), attention.numpy()

def ids_to_words(ids, index_word, end_id):
    """Map decoded ids to words, stopping at <end> or padding."""
    words = []
    for idx in ids:
        idx = int(idx)
        if idx in (0, end_id):
            break
        words.append(index_word.get(idx, '<unk>'))
    return words
