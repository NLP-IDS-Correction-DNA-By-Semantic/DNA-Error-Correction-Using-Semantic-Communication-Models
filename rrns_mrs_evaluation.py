# -*- coding: utf-8 -*-

import random
import warnings
import textwrap
import tempfile
import subprocess
import shutil

from pathlib import Path
from collections import Counter
from functools import reduce
from difflib import SequenceMatcher

import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO, AlignIO
from Bio.Align import PairwiseAligner

from sympy.ntheory.modular import solve_congruence


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/absolute/path/to/project")

FASTA_FILE = BASE_DIR / "sample.fasta"

random.seed(42)

warnings.filterwarnings("ignore")


# ============================================================
# RRNS CONSTANTS
# ============================================================

BASE_MAP = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

INV_MAP = {
    v: k for k, v in BASE_MAP.items()
}

MODULI = [7, 11, 13, 17, 19]

BLOCK_LEN = 12

SEG_LEN = 100

BASES = tuple(BASE_MAP.keys())

OTHER = {
    b: tuple(x for x in BASES if x != b)
    for b in BASES
}


# ============================================================
# FASTA READER
# ============================================================

def read_fasta_sequences(
    file_path,
    segment_len=SEG_LEN
):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found"
        )

    segments = []

    for rec in SeqIO.parse(file_path, "fasta"):

        seq = str(rec.seq).upper()

        if segment_len:

            for i in range(
                0,
                len(seq),
                segment_len
            ):

                chunk = seq[i:i + segment_len]

                if len(chunk) == segment_len:
                    segments.append(chunk)

        else:
            segments.append(seq)

    return segments


# ============================================================
# IDS ERROR SIMULATION
# ============================================================

def introduce_ids_errors(
    seq,
    rates=None
):

    if rates is None:
        rates = {
            "ins": 0.02,
            "del": 0.02,
            "sub": 0.02
        }

    out = []

    i = 0

    r_ins = rates["ins"]
    r_del = rates["del"]
    r_sub = rates["sub"]

    while i < len(seq):

        rnd = random.random()

        if rnd < r_ins:

            out.append(random.choice(BASES))

            out.append(seq[i])

            i += 1

        elif rnd < r_ins + r_del:

            i += 1

        elif rnd < r_ins + r_del + r_sub:

            out.append(
                random.choice(
                    OTHER[seq[i]]
                )
            )

            i += 1

        else:

            out.append(seq[i])

            i += 1

    return "".join(out)


# ============================================================
# DNA <-> NUMERIC CONVERSION
# ============================================================

def to_base4(seq):

    return [
        BASE_MAP[b.upper()]
        for b in seq
    ]


def nums_to_int(block):

    return reduce(
        lambda acc, d: (acc << 2) + d,
        block,
        0
    )


# ============================================================
# RRNS ENCODING
# ============================================================

def rrns_encode_block(
    block_digits,
    moduli=MODULI
):

    n = nums_to_int(block_digits)

    return [
        n % m
        for m in moduli
    ]


def rrns_encode_sequence(
    seq,
    moduli=MODULI,
    block_len=BLOCK_LEN
):

    digits = to_base4(seq)

    blocks = [

        digits[i:i + block_len]

        for i in range(
            0,
            len(digits),
            block_len
        )

        if len(
            digits[i:i + block_len]
        ) == block_len
    ]

    return [
        rrns_encode_block(b, moduli)
        for b in blocks
    ]


# ============================================================
# RRNS DECODING
# ============================================================

def rrns_decode_block(
    residues,
    moduli=MODULI
):

    congr = [

        (residues[i], moduli[i])

        for i in range(len(moduli))
    ]

    try:

        x, _ = solve_congruence(*congr)

        return int(x)

    except ValueError:

        return None


def int_to_dna(
    n,
    length=BLOCK_LEN
):

    digits = [

        (n >> (2 * k)) & 3

        for k in range(length)
    ][::-1]

    return "".join(
        INV_MAP[d]
        for d in digits
    )


# ============================================================
# MULTI READ GENERATION
# ============================================================

def generate_multi_reads(
    sequence,
    num_reads=5,
    rates=None
):

    if rates is None:

        rates = {
            "ins": 0.02,
            "del": 0.02,
            "sub": 0.02
        }

    return [

        introduce_ids_errors(
            sequence,
            rates
        )

        for _ in range(num_reads)
    ]


# ============================================================
# ALIGNMENT
# ============================================================

def align_reads_pairwise(reads):

    if len(reads) < 2:
        return reads

    aligner = PairwiseAligner()

    aligner.mode = "global"

    template = reads[0]

    aligned_reads = [template]

    for read in reads[1:]:

        aln = aligner.align(
            template,
            read
        )[0]

        aligned_reads.append(aln.target)

    return aligned_reads


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_vote(aligned_reads):

    if not aligned_reads:
        return ""

    length = len(aligned_reads[0])

    consensus = []

    for i in range(length):

        column = [

            r[i]

            for r in aligned_reads

            if i < len(r)
            and r[i] != "-"
        ]

        if column:

            common = Counter(
                column
            ).most_common(1)[0][0]

            consensus.append(common)

    return "".join(consensus)


# ============================================================
# SIMILARITY
# ============================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# METRICS
# ============================================================

def edit_distance(a, b):

    return sum(

        1

        for x, y in zip(a, b)

        if x != y

    ) + abs(len(a) - len(b))


def normalized_edit_distance(a, b):

    mxd = max(len(a), len(b))

    if mxd == 0:
        return 0.0

    return edit_distance(a, b) / mxd


def bit_error_rate(
    original,
    predicted
):

    mxd = max(
        len(original),
        len(predicted)
    )

    if mxd == 0:
        return 0.0

    errors = sum(

        1

        for o, p in zip(
            original,
            predicted
        )

        if o != p
    )

    return errors / mxd


# ============================================================
# MARKERS
# ============================================================

LONG_START_MARKER = "ACGTACGTACGT"

LONG_END_MARKER = "GTACGTACGTAC"


def add_markers(
    segment,
    start_marker=LONG_START_MARKER,
    end_marker=LONG_END_MARKER
):

    return (
        f"{start_marker}"
        f"{segment}"
        f"{end_marker}"
    )


def validate_marker_strict(
    seq,
    start_marker=LONG_START_MARKER,
    end_marker=LONG_END_MARKER
):

    seq = seq.upper()

    return (
        seq.startswith(start_marker)
        and
        seq.endswith(end_marker)
    )


def validate_marker_fuzzy(
    sequence,
    start_marker=LONG_START_MARKER,
    end_marker=LONG_END_MARKER,
    threshold=0.85
):

    seq = sequence.upper()

    start = seq[:len(start_marker)]

    end = seq[-len(end_marker):]

    return (

        similarity(
            start,
            start_marker
        ) >= threshold

        and

        similarity(
            end,
            end_marker
        ) >= threshold
    )


# ============================================================
# MSA
# ============================================================

def run_msa(reads):

    if shutil.which("mafft"):

        with tempfile.TemporaryDirectory() as tmp:

            in_fasta = Path(tmp) / "in.fasta"

            out_fasta = Path(tmp) / "out.fasta"

            with in_fasta.open("w") as fh:

                for i, r in enumerate(reads):

                    fh.write(
                        f">read{i}\n{r}\n"
                    )

            cmd = [
                "mafft",
                "--auto",
                "--quiet",
                str(in_fasta)
            ]

            subprocess.run(
                cmd,
                stdout=out_fasta.open("w"),
                check=True
            )

            aln = AlignIO.read(
                out_fasta,
                "fasta"
            )

            return [
                "".join(rec.seq)
                for rec in aln
            ]

    print(
        "MAFFT not found "
        "- using pairwise alignment"
    )

    return align_reads_pairwise(reads)


# ============================================================
# RRNS CONSENSUS DECODING
# ============================================================

def decode_consensus_with_rrns(
    consensus
):

    consensus = consensus.upper()

    orig_len = len(consensus)

    rem = orig_len % BLOCK_LEN

    if rem:

        consensus += (
            "A" * (BLOCK_LEN - rem)
        )

    decoded = []

    for i in range(
        0,
        len(consensus),
        BLOCK_LEN
    ):

        blk = consensus[i:i + BLOCK_LEN]

        residues = rrns_encode_block(
            to_base4(blk)
        )

        x = rrns_decode_block(residues)

        if x is None:
            return None

        decoded.append(
            int_to_dna(x)
        )

    full_decoded = "".join(decoded)

    return full_decoded[:orig_len]


# ============================================================
# PAYLOAD EXTRACTION
# ============================================================

def extract_payload_with_markers(
    seq,
    threshold=0.75
):

    if not validate_marker_fuzzy(
        seq,
        LONG_START_MARKER,
        LONG_END_MARKER,
        threshold
    ):

        return None

    return seq[
        len(LONG_START_MARKER):
        -len(LONG_END_MARKER)
    ]


# ============================================================
# MRS FILTER
# ============================================================

def mrs_filter(
    reads,
    w,
    K=7
):

    v = len(reads)

    if v <= w:
        return reads[:]

    k = len(reads[0])

    Q = [[0] * k for _ in range(v)]

    for j in range(k):

        column = [
            reads[i][j]
            for i in range(v)
        ]

        counts = Counter(column)

        for i in range(v):

            base = reads[i][j]

            if base == '-':
                Q[i][j] = -v

            else:
                Q[i][j] = counts[base]

    q = [
        sum(Q[i]) / k
        for i in range(v)
    ]

    threshold = sorted(
        q,
        reverse=True
    )[w - 1]

    top_idxs = [

        i

        for i, score in enumerate(q)

        if score > threshold
    ]

    ties = [

        i

        for i, score in enumerate(q)

        if score == threshold
    ]

    need = w - len(top_idxs)

    if need > 0 and ties:

        km = k - K + 1

        Qp = {
            i: [0] * km
            for i in ties
        }

        for pos in range(km):

            kmers = [
                reads[i][pos:pos + K]
                for i in ties
            ]

            counts = Counter(kmers)

            for i, mer in zip(ties, kmers):

                Qp[i][pos] = counts[mer]

        qp = {
            i: sum(Qp[i]) / km
            for i in ties
        }

        sel = sorted(
            qp,
            key=lambda i: qp[i],
            reverse=True
        )[:need]

        top_idxs += sel

    return [
        reads[i]
        for i in sorted(top_idxs)
    ]


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    segment,
    num_reads=8,
    mode="rrns+msa",
    ids_rates=None
):

    if ids_rates is None:

        ids_rates = {
            "ins": 0.02,
            "del": 0.02,
            "sub": 0.02
        }

    clean = segment

    if "marker" in mode:
        segment = add_markers(segment)

    reads = generate_multi_reads(
        segment,
        num_reads,
        ids_rates
    )

    if mode.endswith("+mrs"):

        aligned_all = run_msa(reads)

        reads = mrs_filter(
            aligned_all,
            w=4,
            K=7
        )

    aligned = run_msa(reads)

    cons = majority_vote(aligned)

    if "marker" in mode:

        cons = extract_payload_with_markers(
            cons,
            threshold=0.75
        )

        if cons is None:

            return {
                "success": False,
                "reason": "marker lost"
            }

    decoded = decode_consensus_with_rrns(cons)

    if decoded is None:

        return {
            "success": False,
            "reason": "CRT failure"
        }

    return {

        "success": True,

        "BER": bit_error_rate(
            clean,
            decoded
        ),

        "edit": normalized_edit_distance(
            clean,
            decoded
        )
    }


# ============================================================
# SWEEP
# ============================================================

def sweep(
    segments,
    error_rates=(0.02, 0.05),
    modes=(
        "rrns+msa",
        "rrns+marker",
        "rrns+msa+mrs"
    ),
    samples=100
):

    rows = []

    for er in error_rates:

        rate = {
            "ins": er,
            "del": er,
            "sub": er
        }

        for mode in modes:

            for seg in segments[:samples]:

                res = run_pipeline(
                    seg,
                    num_reads=8,
                    mode=mode,
                    ids_rates=rate
                )

                if res.get("success"):

                    rows.append({

                        "mode": mode,

                        "err": er,

                        "BER": res["BER"],

                        "edit": res["edit"]
                    })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading FASTA sequences...")

    segments = read_fasta_sequences(
        FASTA_FILE,
        segment_len=SEG_LEN
    )

    print(
        f"Loaded {len(segments)} segments"
    )

    original = segments[0]

    print(
        "\nOriginal sequence:"
    )

    print(
        textwrap.shorten(
            original,
            width=80
        )
    )


    # ========================================================
    # IDS SIMULATION
    # ========================================================

    corrupted = introduce_ids_errors(
        original
    )

    print("\nCorrupted sequence:")

    print(
        textwrap.shorten(
            corrupted,
            width=80
        )
    )


    # ========================================================
    # RRNS ENCODING
    # ========================================================

    encoded_blocks = rrns_encode_sequence(
        corrupted[:500]
    )

    print(
        "\nFirst RRNS block:"
    )

    print(encoded_blocks[0])


    # ========================================================
    # MULTI READS
    # ========================================================

    multi_reads = generate_multi_reads(
        corrupted[:SEG_LEN],
        num_reads=10
    )

    aligned_reads = align_reads_pairwise(
        multi_reads
    )

    reconstructed = majority_vote(
        aligned_reads
    ).upper()

    sim = similarity(

        original[:len(reconstructed)],

        reconstructed
    )

    print(
        f"\nConsensus similarity:"
        f" {sim:.2%}"
    )


    # ========================================================
    # RRNS DECODING
    # ========================================================

    decoded_int = rrns_decode_block(
        encoded_blocks[0]
    )

    decoded_dna = (

        int_to_dna(decoded_int)

        if decoded_int is not None

        else "ERR"
    )

    print("\nDecoded block:")

    print(decoded_dna)


    # ========================================================
    # PIPELINE TESTS
    # ========================================================

    print("\nPipeline tests:\n")

    print(
        run_pipeline(
            segments[0],
            mode="rrns+msa"
        )
    )

    print(
        run_pipeline(
            segments[0],
            mode="rrns+marker"
        )
    )

    print(
        run_pipeline(
            segments[0],
            mode="rrns+msa+mrs"
        )
    )


    # ========================================================
    # SWEEP
    # ========================================================

    print("\nRunning sweep...")

    df = sweep(

        segments,

        error_rates=(0.02, 0.05),

        samples=50
    )

    summary = (

        df

        .groupby(["mode", "err"])

        .BER

        .agg(
            mean="mean",
            std="std",
            count="count"
        )

        .reset_index()
    )


    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    for mode, sub in summary.groupby("mode"):

        sub = sub.sort_values("err")

        ax.errorbar(

            sub["err"],

            sub["mean"],

            yerr=sub["std"],

            marker="o",

            label=mode,

            capsize=3
        )

    ax.set_xlabel("IDS error rate")

    ax.set_ylabel("Mean BER ± 1σ")

    ax.set_title(
        "RRNS-based Methods"
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


    # ========================================================
    # PARAMETER SWEEP
    # ========================================================

    print("\nParameter sweep...")

    block_lengths = [8, 10, 12]

    moduli_sets = [

        [5, 7, 11, 13, 17],

        [3, 5, 7, 11, 13],

        [7, 11, 13, 17, 19]
    ]

    error_rates = [0.02, 0.05]

    modes = [
        "rrns+msa",
        "rrns+marker",
        "rrns+msa+mrs"
    ]

    results = []

    global BLOCK_LEN
    global MODULI

    for bl in block_lengths:

        for mods in moduli_sets:

            BLOCK_LEN = bl

            MODULI = mods

            for er in error_rates:

                rates = {
                    "ins": er,
                    "del": er,
                    "sub": er
                }

                for mode in modes:

                    res = run_pipeline(
                        segments[0],
                        num_reads=8,
                        mode=mode,
                        ids_rates=rates
                    )

                    if res.get("success"):

                        results.append({

                            "BLOCK_LEN": bl,

                            "MODULI": ",".join(
                                map(str, mods)
                            ),

                            "mode": mode,

                            "err": er,

                            "BER": res["BER"],

                            "edit": res["edit"]
                        })

    df_full = pd.DataFrame(results)

    pivot = df_full.pivot_table(

        index=[
            "BLOCK_LEN",
            "MODULI",
            "mode"
        ],

        columns="err",

        values="BER",

        aggfunc="mean"
    )

    print("\nParameter Sweep Results:\n")

    print(pivot)

    print("\nPipeline completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()