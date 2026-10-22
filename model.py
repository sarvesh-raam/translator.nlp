import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # mask shape == (batch_size, max_length); False on padding (id 0)
        mask = tf.math.not_equal(x, 0)
        x = self.embedding(x)
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
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, enc_mask=None):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output, enc_mask)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab_size)
        x = self.fc(output)

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

def ids_to_words(ids, index_word, end_id):
    """Map decoded ids to words, stopping at <end> or padding."""
    words = []
    for idx in ids:
        idx = int(idx)
        if idx in (0, end_id):
            break
        words.append(index_word.get(idx, '<unk>'))
    return words
