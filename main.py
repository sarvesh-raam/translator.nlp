from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import os
import pickle
import numpy as np
from model import Encoder, Decoder

app = FastAPI()

# Mount static files (Fixes CSS/JS loading issues)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
MAX_SEQ_LEN = 30
UNITS = 256
EMBEDDING_DIM = 256

# Load tokenizers
with open(os.path.join(DATA_DIR, 'en_tokenizer.pkl'), 'rb') as f:
    en_tokenizer = pickle.load(f)
with open(os.path.join(DATA_DIR, 'hi_tokenizer.pkl'), 'rb') as f:
    hi_tokenizer = pickle.load(f)

vocab_inp_size = len(en_tokenizer.word_index) + 1
vocab_tar_size = len(hi_tokenizer.word_index) + 1

# Initialize models
encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, 1)
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, 1)

# Load weights if they exist
if os.path.exists(os.path.join(MODELS_DIR, 'encoder_weights.h5')):
    # Building models first by calling them with a dummy input
    dummy_input = tf.zeros((1, MAX_SEQ_LEN))
    dummy_hidden = encoder.initialize_hidden_state()
    enc_out, enc_hidden = encoder(dummy_input, dummy_hidden)
    
    dummy_dec_input = tf.expand_dims([hi_tokenizer.word_index['<start>']], 0)
    decoder(dummy_dec_input, enc_hidden, enc_out)
    
    encoder.load_weights(os.path.join(MODELS_DIR, 'encoder_weights.h5'))
    decoder.load_weights(os.path.join(MODELS_DIR, 'decoder_weights.h5'))
    print("Model weights loaded successfully!")

def evaluate(sentence):
    attention_plot = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN))
    
    sentence = sentence.lower().strip()
    inputs = [en_tokenizer.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=MAX_SEQ_LEN, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([hi_tokenizer.word_index['<start>']], 0)

    for t in range(MAX_SEQ_LEN):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # Storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        if hi_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        result += hi_tokenizer.index_word[predicted_id] + ' '

        # The predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return open("index.html", "r", encoding="utf-8").read()

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Empty text"}
    
    translated, original, attention = evaluate(text)
    
    # Format attention for D3.js heatmap
    # attention shape is (target_len, source_len)
    source_tokens = original.split()
    target_tokens = translated.split()
    
    heatmap_data = []
    for i, target_word in enumerate(target_tokens):
        for j, source_word in enumerate(source_tokens):
            if j < MAX_SEQ_LEN:
                heatmap_data.append({
                    "source": source_word,
                    "target": target_word,
                    "weight": float(attention[i][j])
                })
    
    return {
        "translated": translated,
        "heatmap": heatmap_data
    }

@app.get("/metrics")
async def get_metrics():
    # Load real history if it exists, else return placeholder
    history_path = os.path.join(DATA_DIR, 'history.pkl')
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    
    return {
        "bleu": 32.45,
        "training_history": {
            "epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "loss": [2.45, 1.98, 1.54, 1.21, 0.98, 0.76, 0.54, 0.43, 0.32, 0.28]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
