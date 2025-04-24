from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from Levenshtein import distance as levenshtein_distance
import os

def load_sequence(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip().upper()

def compute_bleu(reference, candidate):
    ref_tokens = list(reference)
    cand_tokens = list(candidate)
    return sentence_bleu([ref_tokens], cand_tokens)

def compute_ber(reference, candidate):
    min_len = min(len(reference), len(candidate))
    errors = sum(1 for a, b in zip(reference, candidate) if a != b)
    return errors / min_len

def compute_levenshtein(reference, candidate):
    lev_dist = levenshtein_distance(reference, candidate)
    normalized = lev_dist / len(reference)
    return lev_dist, normalized

def compute_bert_score(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en", model_type="bert-base-uncased")
    return P.item(), R.item(), F1.item()

if __name__ == "__main__":
    original_path = "data/original.txt"
    distorted_path = "data/distorted.txt"  # Or use corrected.txt

    assert os.path.exists(original_path), "Original file not found"
    assert os.path.exists(distorted_path), "Distorted file not found"

    original = load_sequence(original_path)
    distorted = load_sequence(distorted_path)

    print("\n📊 Evaluation Results:")
    
    bleu = compute_bleu(original, distorted)
    print(f"🟦 BLEU Score         : {bleu:.4f}")
    
    ber = compute_ber(original, distorted)
    print(f"🧮 Bit Error Rate     : {ber:.4f}")
    
    lev_raw, lev_norm = compute_levenshtein(original, distorted)
    print(f"✏️  Levenshtein Dist.  : {lev_raw} (Normalized: {lev_norm:.4f})")
    
    P, R, F1 = compute_bert_score(original, distorted)
    print(f"⚠️  BERTScore F1       : {F1:.4f} (P={P:.4f}, R={R:.4f})")
    print("⚠️  Note: BERTScore is optimized for natural language, not DNA.")
