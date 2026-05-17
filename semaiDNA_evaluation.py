# -*- coding: utf-8 -*-

import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein
import tensorflow as tf
import torch

from tqdm import tqdm
from Bio import SeqIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/absolute/path/to/NLP_Project")

INPUT_FASTA = BASE_DIR / "chr1_100kb.fasta"
REFGENE_CSV = BASE_DIR / "refgene_table.csv"

ENRICHED_FASTA = BASE_DIR / "enriched_ucsc_fasta.fa"

DATASET_CSV = BASE_DIR / "semai_dna_dataset.csv"
NOISY_DATASET_CSV = BASE_DIR / "semai_dna_noisy_kmer_dataset.csv"

K = 6
MAXLEN = 512
NUM_READS = 5

DNABERT_MODEL = "zhihan1996/DNA_bert_6"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD DNABERT
# ============================================================

tokenizer_dnabert = AutoTokenizer.from_pretrained(
    DNABERT_MODEL
)

dnabert_model = AutoModel.from_pretrained(
    DNABERT_MODEL
)

dnabert_model = dnabert_model.to(DEVICE)

dnabert_model.eval()


# ============================================================
# ENRICH FASTA HEADERS
# ============================================================

print("\nLoading refgene table...")

refgene_df = pd.read_csv(REFGENE_CSV)

assert 'name' in refgene_df.columns
assert 'name2' in refgene_df.columns

refseq_to_gene = dict(
    zip(
        refgene_df['name'],
        refgene_df['name2']
    )
)

records = []
gene_names = []

print("Enriching FASTA headers...")

for record in SeqIO.parse(INPUT_FASTA, "fasta"):

    full_id = record.id.strip()

    base_id = full_id.split('.')[0]

    gene_name = refseq_to_gene.get(
        base_id,
        "UNKNOWN"
    )

    record.description = f"{full_id} {gene_name}"

    records.append(record)

    gene_names.append(gene_name)

SeqIO.write(
    records,
    ENRICHED_FASTA,
    "fasta"
)

print(f"Saved enriched FASTA -> {ENRICHED_FASTA}")


# ============================================================
# CREATE LABELED DATASET
# ============================================================

data = [
    {
        "id": r.id,
        "sequence": str(r.seq),
        "gene": g
    }
    for r, g in zip(records, gene_names)
]

df_genes = pd.DataFrame(data)

df_genes = df_genes[
    df_genes["gene"] != "UNKNOWN"
]

df_genes.to_csv(DATASET_CSV, index=False)

print(f"Saved dataset -> {DATASET_CSV}")


# ============================================================
# FILTER CLEAN DNA SEQUENCES
# ============================================================

sequences = []
labels = []

for record in SeqIO.parse(ENRICHED_FASTA, "fasta"):

    dna_seq = str(record.seq).upper().replace("N", "")

    if len(dna_seq) < 30:
        continue

    parts = record.description.split()

    label = parts[1] if len(parts) > 1 else "UNKNOWN"

    if label == "UNKNOWN":
        continue

    sequences.append(dna_seq)

    labels.append(label)

df = pd.DataFrame({
    "sequence": sequences,
    "label": labels
})

df.drop_duplicates(inplace=True)

df.to_csv(DATASET_CSV, index=False)

print(f"Saved filtered dataset -> {DATASET_CSV}")


# ============================================================
# DNABERT EMBEDDING FUNCTIONS
# ============================================================

def dnabert_embed(seq, k=6):

    if len(seq) < k:
        raise ValueError("Sequence too short")

    kmer_seq = ' '.join([
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
    ])

    inputs = tokenizer_dnabert(
        kmer_seq,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = dnabert_model(**inputs)

    return outputs.last_hidden_state[:, 0, :]\
        .squeeze()\
        .cpu()\
        .numpy()


def dnabert_embed_batch(seq_list, k=6):

    kmers_list = [
        ' '.join([
            seq[i:i+k]
            for i in range(len(seq) - k + 1)
        ])
        for seq in seq_list
    ]

    inputs = tokenizer_dnabert(
        kmers_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = dnabert_model(**inputs)

    return outputs.last_hidden_state[:, 0, :]\
        .cpu()\
        .numpy()


# ============================================================
# IDS CORRUPTION
# ============================================================

def simulate_ids_errors(
    seq,
    ins_prob=0.005,
    del_prob=0.005,
    sub_prob=0.01
):

    bases = ['A', 'C', 'G', 'T']

    result = []

    i = 0

    while i < len(seq):

        r = random.random()

        if r < del_prob:
            i += 1
            continue

        elif r < del_prob + sub_prob:

            result.append(
                random.choice([
                    b for b in bases
                    if b != seq[i]
                ])
            )

        else:
            result.append(seq[i])

        if random.random() < ins_prob:
            result.append(random.choice(bases))

        i += 1

    return ''.join(result)


def tokenize_kmers(seq, k=6):

    return [
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
    ]


def normalized_edit_distance(a, b):

    max_len = max(len(a), len(b))

    if max_len == 0:
        return 0.0

    return (
        Levenshtein.distance(a, b)
        / max_len
    )


# ============================================================
# GENERATE NOISY DATASET
# ============================================================

print("\nGenerating noisy dataset...")

noisy_sequences = []
tokenized_kmers = []
edit_distances = []
cosine_similarities = []
clean_sequences = []
labels_clean = []

skipped = 0

for seq, label in tqdm(
    zip(df['sequence'], df['label']),
    total=len(df)
):

    if len(seq) < K:
        skipped += 1
        continue

    try:
        clean_emb = dnabert_embed(seq).reshape(1, -1)

    except Exception:
        skipped += 1
        continue

    noisy_reads = []

    for _ in range(NUM_READS * 2):

        r = simulate_ids_errors(seq)

        if len(r) >= K:
            noisy_reads.append(r)

        if len(noisy_reads) >= NUM_READS:
            break

    if not noisy_reads:
        skipped += 1
        continue

    try:
        noisy_matrix = dnabert_embed_batch(
            noisy_reads,
            k=K
        )

    except Exception:
        skipped += 1
        continue

    sim_scores = cosine_similarity(
        clean_emb,
        noisy_matrix
    )[0]

    best_idx = np.argmax(sim_scores)

    best_read = noisy_reads[best_idx]

    best_sim = sim_scores[best_idx]

    dist = normalized_edit_distance(
        seq,
        best_read
    )

    kmers = tokenize_kmers(best_read, k=K)

    noisy_sequences.append(" ".join(kmers))

    tokenized_kmers.append(kmers)

    edit_distances.append(dist)

    cosine_similarities.append(best_sim)

    clean_sequences.append(seq)

    labels_clean.append(label)


df_clean = pd.DataFrame({
    "sequence": clean_sequences,
    "noisy_sequence": noisy_sequences,
    "kmer_tokens": tokenized_kmers,
    "edit_distance": edit_distances,
    "cosine_similarity": cosine_similarities,
    "label": labels_clean
})

df_clean.to_csv(NOISY_DATASET_CSV, index=False)

print(f"Saved noisy dataset -> {NOISY_DATASET_CSV}")

print(f"Skipped {skipped} sequences")


# ============================================================
# IDS STATISTICS PLOTS
# ============================================================

print("\nIDS Corruption Statistics")

print(
    f"Mean Edit Distance: "
    f"{np.mean(edit_distances)*100:.2f}%"
)

print(
    f"Mean Cosine Similarity: "
    f"{np.mean(cosine_similarities):.3f}"
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.hist(
    edit_distances,
    bins=30
)

plt.title("Edit Distance Distribution")

plt.xlabel("Normalized Edit Distance")

plt.ylabel("Count")


plt.subplot(1, 2, 2)

plt.hist(
    cosine_similarities,
    bins=30
)

plt.title("Cosine Similarity Distribution")

plt.xlabel("Cosine Similarity")

plt.ylabel("Count")

plt.tight_layout()

plt.show()


# ============================================================
# TOKENIZATION
# ============================================================

tokenizer = Tokenizer(
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    df_clean['noisy_sequence']
)

sequences = tokenizer.texts_to_sequences(
    df_clean['noisy_sequence']
)

X = pad_sequences(
    sequences,
    maxlen=MAXLEN,
    padding='post'
)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df_clean['label']
)

num_classes = len(label_encoder.classes_)

y_cat = to_categorical(y)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)


# ============================================================
# CNN MODEL
# ============================================================

print("\nTraining CNN model...")

vocab_size = len(tokenizer.word_index) + 1

cnn_model = models.Sequential([

    layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=MAXLEN
    ),

    layers.Conv1D(
        64,
        5,
        activation='relu'
    ),

    layers.GlobalMaxPooling1D(),

    layers.Dense(
        64,
        activation='relu'
    ),

    layers.Dense(
        num_classes,
        activation='softmax'
    )
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = cnn_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)


# ============================================================
# CNN EVALUATION
# ============================================================

loss, acc = cnn_model.evaluate(
    X_test,
    y_test
)

print(f"\nCNN Test Accuracy: {acc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

y_pred = cnn_model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(xticks_rotation=90)

plt.title("CNN Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# SAVE CNN MODEL
# ============================================================

cnn_model.save(
    BASE_DIR / "semai_dna_cnn_model.keras"
)

with open(
    BASE_DIR / "semai_dna_tokenizer.pkl",
    "wb"
) as f:

    pickle.dump(tokenizer, f)

with open(
    BASE_DIR / "semai_dna_label_encoder.pkl",
    "wb"
) as f:

    pickle.dump(label_encoder, f)

print("\nSaved CNN model and preprocessors.")


# ============================================================
# BILSTM MODEL
# ============================================================

print("\nTraining BiLSTM model...")

bilstm_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        )
    ),

    tf.keras.layers.GlobalMaxPooling1D(),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )
])

bilstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

bilstm_model.summary()


history = bilstm_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)


# ============================================================
# BILSTM EVALUATION
# ============================================================

loss, acc = bilstm_model.evaluate(
    X_test,
    y_test
)

print(f"\nBiLSTM Test Accuracy: {acc:.4f}")


# ============================================================
# BILSTM CONFUSION MATRIX
# ============================================================

y_pred = bilstm_model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(xticks_rotation=90)

plt.title("BiLSTM Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# SAMPLE TESTING
# ============================================================

def test_sample(idx=None, num_reads=5, k=6):

    print("=" * 60)

    if idx is None:
        idx = np.random.randint(len(X_test))

    y_true_class = np.argmax(y_test[idx])

    true_label = label_encoder.inverse_transform(
        [y_true_class]
    )[0]

    clean_seq = df_clean.iloc[idx]["sequence"]

    print(f"Sample #{idx}")

    print(f"True Label: {true_label}")

    noisy_reads = [
        simulate_ids_errors(clean_seq)
        for _ in range(num_reads)
    ]

    best_sim = -1

    best_read = ""

    best_pred_label = ""

    for r in noisy_reads:

        try:

            kmers = [
                r[i:i+k]
                for i in range(len(r) - k + 1)
            ]

            kmer_string = " ".join(kmers)

            seq_encoded = tokenizer.texts_to_sequences(
                [kmer_string]
            )

            X_input = pad_sequences(
                seq_encoded,
                maxlen=MAXLEN,
                padding='post'
            )

            pred_probs = bilstm_model.predict(X_input)

            pred_class = np.argmax(pred_probs)

            pred_label = label_encoder.inverse_transform(
                [pred_class]
            )[0]

            sim = cosine_similarity(
                [dnabert_embed(clean_seq)],
                [dnabert_embed(r)]
            )[0][0]

            print(
                f"Prediction: {pred_label} | "
                f"Similarity: {sim:.4f}"
            )

            if sim > best_sim:

                best_sim = sim

                best_read = r

                best_pred_label = pred_label

        except Exception as e:

            print(f"Skipped read: {e}")

            continue

    print("\nBest Prediction")

    print(f"Predicted Label: {best_pred_label}")

    print(
        f"Correct? "
        f"{'YES' if best_pred_label == true_label else 'NO'}"
    )

    print(f"Similarity: {best_sim:.4f}")

    print("=" * 60)


test_sample()

test_sample(idx=42)


# ============================================================
# SEQUENCE LENGTH DISTRIBUTION
# ============================================================

lengths = [
    len(seq)
    for seq in df_clean['sequence']
]

plt.hist(lengths, bins=50)

plt.title("Sequence Length Distribution")

plt.xlabel("Length")

plt.ylabel("Count")

plt.show()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

y_pred_probs = cnn_model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# HEATMAP CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.show()


# ============================================================
# DNABERT EMBEDDINGS EXPORT
# ============================================================

print("\nGenerating DNABERT embeddings...")

def to_kmers(seq, k=6):

    return " ".join([
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
    ])


def get_dnabert6_embedding(seq):

    kmer_seq = to_kmers(seq.upper())

    inputs = tokenizer_dnabert(
        kmer_seq,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        outputs = dnabert_model(**inputs)

        cls_vec = outputs.last_hidden_state[:, 0, :]

    return cls_vec.squeeze().cpu().numpy()


embeddings = []

embedding_labels = []

for seq, label in zip(
    df_clean['sequence'],
    df_clean['label']
):

    try:

        vec = get_dnabert6_embedding(seq[:300])

        embeddings.append(vec)

        embedding_labels.append(label)

    except Exception as e:

        print(f"Skipped: {e}")


np.save(
    BASE_DIR / "dnabert6_embeddings.npy",
    np.stack(embeddings)
)

pd.Series(embedding_labels).to_csv(
    BASE_DIR / "dnabert6_labels.csv",
    index=False
)

print("\nPipeline completed successfully.")