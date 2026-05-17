# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
import glob
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch

from Bio import Entrez, SeqIO


# CONFIGURATION

BASE_DIR = Path("/absolute/path/to/NLP_Project")

DEEPJSOC_DIR = BASE_DIR / "deepjsoc"
DNA_JSOC_DIR = BASE_DIR / "dna_jsoc"

HUMAN_FASTA_GZ = "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
HUMAN_FASTA = "Homo_sapiens.GRCh38.dna.primary_assembly.fa"

CHIMP_FASTA = BASE_DIR / "sequence.fasta"

SEQ_LEN = 256

Entrez.email = "your_email@example.com"

def run_command(command):
    """
    Run shell command and stop on failure.
    """
    print(f"\nRunning:\n{command}\n")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{command}")


def load_seq(fasta_path):
    """
    Read FASTA and return concatenated uppercase sequence.
    """
    return ''.join(
        str(rec.seq).upper()
        for rec in SeqIO.parse(fasta_path, 'fasta')
    )

DNA_JSOC_DIR.mkdir(exist_ok=True, parents=True)


# HUMAN GENOME
if not (DEEPJSOC_DIR / HUMAN_FASTA).exists():

    os.chdir(DEEPJSOC_DIR)

    run_command(
        "wget ftp://ftp.ensembl.org/pub/release-108/fasta/"
        "homo_sapiens/dna/"
        "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    )

    run_command(f"gunzip {HUMAN_FASTA_GZ}")


# HUMAN MITOCHONDRIAL GENOME
mt_id = "NC_012920.1"
out_file = DNA_JSOC_DIR / "human_mt.fasta"

with Entrez.efetch(
    db="nucleotide",
    id=mt_id,
    rettype="fasta",
    retmode="text"
) as handle:

    seq_record = SeqIO.read(handle, "fasta")

SeqIO.write(seq_record, out_file, "fasta")

print(
    f"Downloaded mitochondrial genome "
    f"({len(seq_record.seq)} bp)"
)


# TRAIN ESTIMATOR
os.chdir(DEEPJSOC_DIR)

train_estimator_cmd = f"""
python train_estimator.py \
    --channel_bits 128 \
    --Nc 2 \
    --d_rnn 256 \
    --bs 64 \
    --epochs 6 \
    --step 200 \
    --training_Pd 0.03 \
    --training_Ps 0.03 \
    --training_Pi 0.03
"""

run_command(train_estimator_cmd)

src = glob.glob(
    str(DEEPJSOC_DIR / "*estimator*.pth")
)[0]

dst = DNA_JSOC_DIR / "estimator.pt"

shutil.copy(src, dst)

print(f"Estimator saved -> {dst}")


# CREATE DATASET (CHIMP + HUMAN)
chimp_seq = load_seq(CHIMP_FASTA)

human_seq = load_seq(
    DEEPJSOC_DIR / HUMAN_FASTA
)

combined_seq = chimp_seq + human_seq

print(
    f"Total length:\n"
    f"Chimp: {len(chimp_seq)}\n"
    f"Human: {len(human_seq)}\n"
    f"Combined: {len(combined_seq)}"
)

chunks = [
    combined_seq[i:i + SEQ_LEN]
    for i in range(
        0,
        len(combined_seq) - SEQ_LEN + 1,
        SEQ_LEN
    )
]

random.shuffle(chunks)

a = int(0.8 * len(chunks))
b = int(0.9 * len(chunks))

splits = {
    'train': chunks[:a],
    'val': chunks[a:b],
    'test': chunks[b:]
}

for split, data in splits.items():

    out_path = DNA_JSOC_DIR / f"chimp{SEQ_LEN}.{split}"

    with open(out_path, 'w') as f:
        f.write('\n'.join(data))

    print(f"Wrote {len(data)} sequences -> {out_path}")


# CREATE VOCAB
tok2idx = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<MASK>": 3,
    "<UNK>": 4,
    "A": 5,
    "C": 6,
    "G": 7,
    "T": 8,
    "N": 9
}

idx2tok = [
    k for k, _ in sorted(
        tok2idx.items(),
        key=lambda kv: kv[1]
    )
]

vocab = {
    "token_to_idx": tok2idx,
    "idx_to_token": idx2tok
}

vocab_path = DNA_JSOC_DIR / "vocab.json"

with open(vocab_path, 'w') as f:
    json.dump(vocab, f, indent=2)

print("Vocabulary created.")

ckpt = torch.load(dst)

w = ckpt['bir_layers.0.weight_ih_l0']

print(
    "actual_size in checkpoint =",
    w.shape[1]
)


# TRAIN MAIN MODEL
main_cmd = f"""
python main.py \
    --vocab-file {vocab_path} \
    --batch-size 4 \
    --epochs 6 \
    --warm-start -1 \
    --safety-len 360 \
    --marker-enc-size 360 \
    --channel-in-len 36 \
    --estimator-file {dst}
"""

run_command(main_cmd)


# PERFORMANCE EVALUATION
performance_cmd = f"""
python performance.py \
    --vocab-file {vocab_path} \
    --checkpoint-path checkpoints \
    --batch-size 8 \
    --ps 0.03 \
    --pd-list 0.0 0.02 0.05 0.08 \
    --estimator-file {dst}
"""

run_command(performance_cmd)

results_path = DNA_JSOC_DIR / "results.csv"

df = pd.read_csv(results_path)

numeric_cols = [
    'Pd',
    'BLEU-1',
    'DNABERT-sim',
    'BERT-sim',
    'Edit-Distance',
    'Token-Accuracy',
    'Sequence-Accuracy'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

print("\n===== Performance Metrics =====")

print(
    f"{'Pd':<8} "
    f"{'BLEU-1':<10} "
    f"{'EditDist':<12}"
)

print("=" * 40)

for _, row in df.iterrows():

    print(
        f"{row['Pd']:<8.3f} "
        f"{row['BLEU-1']:<10.4f} "
        f"{row['Edit-Distance']:<12.2f}"
    )

plt.figure(figsize=(10, 6))

plt.plot(
    df['Pd'],
    df['BLEU-1'],
    marker='o',
    label='BLEU-1'
)

plt.xlabel(
    'Insertion / Deletion / Substitution probability (Pd)'
)

plt.ylabel('BLEU Score')

plt.title('BLEU-1 vs Pd')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

if 'BERT-sim' in df.columns:

    plt.figure(figsize=(10, 6))

    plt.plot(
        df['Pd'],
        df['BLEU-1'],
        marker='o',
        label='BLEU-1'
    )

    plt.plot(
        df['Pd'],
        df['BERT-sim'],
        marker='o',
        label='BERT-sim'
    )

    plt.xlabel(
        'Insertion / Deletion / Substitution probability (Pd)'
    )

    plt.ylabel('Score')

    plt.title(
        'BLEU-1 vs BERT-sim across Pd levels'
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


print("\nPipeline completed successfully.")