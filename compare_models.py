"""Scores this model and a pretrained baseline (Helsinki-NLP/opus-mt-en-hi) on the same IITB test
sentences, with the same references and the same output cleaning, and saves every translation.

    pip install torch transformers sentencepiece   (on top of requirements.txt)
    python compare_models.py
"""
import json
import os
import sacrebleu
from config import DATA_DIR, MAX_SEQ_LEN
from inference import Translator
from preprocess import clean_text

BASELINE = "Helsinki-NLP/opus-mt-en-hi"
BATCH_SIZE = 16

def load_test_set():
    """The official IITB test pairs kept by preprocess.clean_pairs, with the raw English alongside:
    the baseline was trained on cased, untokenised text, so it gets the sentence as written."""
    from datasets import load_dataset
    rows = load_dataset("cfilt/iitb-english-hindi", split="test")['translation']
    raw_en, clean_en, refs = [], [], []
    for row in rows:
        en = clean_text(row['en'])
        hi = clean_text(row['hi'], is_hindi=True)
        if 3 <= len(en.split()) <= MAX_SEQ_LEN and 0 < len(hi.split()) <= MAX_SEQ_LEN - 2:
            raw_en.append(row['en'].strip())
            clean_en.append(en)
            refs.append(hi)

    # Must be exactly the sentences train.py reported the test BLEU on
    with open(os.path.join(DATA_DIR, 'test_pairs.json'), encoding='utf-8') as f:
        saved = json.load(f)
    assert clean_en == saved['en'] and refs == saved['hi'], "test set differs from data/test_pairs.json"
    return raw_en, clean_en, refs

def translate_baseline(sentences):
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    tokenizer = MarianTokenizer.from_pretrained(BASELINE)
    model = MarianMTModel.from_pretrained(BASELINE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = tokenizer(sentences[i:i + BATCH_SIZE], return_tensors="pt", padding=True)
            generated = model.generate(**batch, num_beams=5, max_new_tokens=128)
            out += tokenizer.batch_decode(generated, skip_special_tokens=True)
            print(f"  baseline {len(out)}/{len(sentences)}", end="\r")
    print()
    return out

def score(hypotheses, refs):
    return {"bleu": round(sacrebleu.corpus_bleu(hypotheses, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hypotheses, [refs]).score, 2)}

def main():
    raw_en, clean_en, refs = load_test_set()
    print(f"{len(refs)} test sentences")

    translator = Translator()
    ours = [translator.translate(en)[0] for en in clean_en]
    # Same cleaning as the references (lowercase, padded punctuation, Devanagari only)
    baseline = [clean_text(hi, is_hindi=True) for hi in translate_baseline(raw_en)]

    results = {"test_sentences": len(refs),
               "ours": score(ours, refs),
               "baseline": {"model": BASELINE, **score(baseline, refs)}}
    print(json.dumps(results, indent=2))

    results["translations"] = [{"en": en, "reference": ref, "ours": o, "baseline": b}
                               for en, ref, o, b in zip(raw_en, refs, ours, baseline)]
    with open(os.path.join(DATA_DIR, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
