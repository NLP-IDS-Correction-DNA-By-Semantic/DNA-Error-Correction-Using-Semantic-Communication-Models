from pathlib import Path
import random

def extract_chr_sequence(filename, chr_name="chr1", length=10000, offset=1_000_000):
    """
    Extracts `length` base pairs from `chr_name` starting at `offset`.
    Skips 'N' regions.
    """
    seq = []
    reading = False
    total_seen = 0

    with open(filename, "r") as f:
        for line in f:
            if line.startswith(">"):
                reading = line.strip().startswith(f">{chr_name}")
                continue
            if reading:
                line_seq = line.strip().upper()
                total_seen += len(line_seq)
                if total_seen < offset:
                    continue
                for base in line_seq:
                    if base in ["A", "C", "G", "T"]:
                        seq.append(base)
                    if len(seq) >= length:
                        return ''.join(seq)

    return ''.join(seq)


def simulate_ids(sequence, p_insert=0.03, p_delete=0.03, p_substitute=0.03):
    dna_bases = ['A', 'C', 'G', 'T']
    noisy_seq = []
    for base in sequence:
        r = random.random()
        if r < p_delete:
            continue
        if r < p_delete + p_substitute:
            noisy_seq.append(random.choice([b for b in dna_bases if b != base]))
            continue
        if r < p_delete + p_substitute + p_insert:
            noisy_seq.append(base)
            noisy_seq.append(random.choice(dna_bases))
            continue
        noisy_seq.append(base)
    return ''.join(noisy_seq)

def save_sequence(seq, filename):
    with open(filename, 'w') as f:
        f.write(seq)

# Run this
if __name__ == "__main__":
    fasta_path = "hg38.fa"
    chr_seq = extract_chr_sequence(fasta_path, "chr1", 10000)

    save_sequence(chr_seq, "data/original.txt")

    distorted_seq = simulate_ids(chr_seq, p_insert=0.03, p_delete=0.03, p_substitute=0.03)
    save_sequence(distorted_seq, "data/distorted.txt")

    print("✅ Done: original.txt and distorted.txt saved in /data")
