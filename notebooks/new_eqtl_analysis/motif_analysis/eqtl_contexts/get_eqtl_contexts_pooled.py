import os
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from Bio import Seq, SeqIO
from pyfaidx import Fasta


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--eqtl_tsv_path", type=str, default="../../merged_eqtl_results.tsv")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg38/hg38.fa")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[51, 101])
    return parser.parse_args()


def read_eqtl_df(eqtl_tsv_path: str) -> pd.DataFrame:
    eqtl_df = pd.read_csv(eqtl_tsv_path, sep="\t", header=0, index_col=0)
    eqtl_df = eqtl_df[eqtl_df["finemapped"]].copy()
    
    # Map n_correct_CAGE_SAD to ["consistently_correct", "consistently_incorrect", "inconsistent"]
    category_map = {5: "consistently_correct", 0: "consistently_incorrect"}
    eqtl_df["category"] = eqtl_df["n_correct_CAGE_SAD"].map(category_map).fillna("inconsistent")
    return eqtl_df


def publish_fasta(output_path: str, sequences: list, variants: list, intervals: list):
    assert len(sequences) == len(variants) == len(intervals)
    assert len(set(intervals)) == len(intervals)

    seq_records = [
        SeqIO.SeqRecord(Seq.Seq(seq), id=interval, description=variant) 
        for (seq, interval, variant) in zip(sequences, intervals, variants)
    ]
    SeqIO.write(seq_records, output_path, "fasta")


def get_sequences(variants: list, context_size: int, fasta: Fasta):
    sequences, intervals = [], []
    for variant in variants:
        chrom, variant_pos, ref, *_ = variant.split("_")
        variant_pos = int(variant_pos)
        start_pos = variant_pos - context_size // 2
        end_pos = start_pos + context_size - 1
        seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()
        assert len(seq) == context_size
        assert seq[context_size // 2] == ref.upper()
        sequences.append(seq)
        intervals.append(f"{chrom}:{start_pos}-{end_pos}")
    return sequences, intervals


def get_eqtls_for_comparison(
    eqtl_df: pd.DataFrame, 
    category1: str,
    category2: str,
):
    cat1_eqtl_df = eqtl_df[eqtl_df["category"] == category1].copy()
    cat2_eqtl_df = eqtl_df[eqtl_df["category"] == category2].copy()
    print(f"Original sizes: {category1}={cat1_eqtl_df.shape[0]}, {category2}={cat2_eqtl_df.shape[0]}")

    # Keep a single row per variant in case the variant is present in multiple tissues
    cat1_eqtl_df = cat1_eqtl_df.groupby("variant").first().reset_index()
    cat2_eqtl_df = cat2_eqtl_df.groupby("variant").first().reset_index()
    print(f"Unique sizes: {category1}={cat1_eqtl_df.shape[0]}, {category2}={cat2_eqtl_df.shape[0]}")

    # Subset input_eqtl_df and background_eqtl_df so that both contain an equal number of variants
    # per tissue
    cat1_sampled_dfs, cat2_sampled_dfs = [], []
    for tissue in set(cat1_eqtl_df["tissue"]) | set(cat2_eqtl_df["tissue"]):
        cat1_tissue_df = cat1_eqtl_df[cat1_eqtl_df["tissue"] == tissue]
        cat2_tissue_df = cat2_eqtl_df[cat2_eqtl_df["tissue"] == tissue]
        n_variants = min(cat1_tissue_df.shape[0], cat2_tissue_df.shape[0])
        cat1_sampled_dfs.append(cat1_tissue_df.sample(n_variants))
        cat2_sampled_dfs.append(cat2_tissue_df.sample(n_variants))

    cat1_eqtl_df = pd.concat(cat1_sampled_dfs)
    cat2_eqtl_df = pd.concat(cat2_sampled_dfs)

    # Get sequences
    cat1_variants = cat1_eqtl_df["variant"].tolist()
    cat2_variants = cat2_eqtl_df["variant"].tolist()
    print(f"Final sizes: {category1}={len(cat1_variants)}, {category2}={len(cat2_variants)}")
    print("")
    return cat1_variants, cat2_variants


def generate_fastas_for_comparison(
    eqtl_df: pd.DataFrame,
    fasta: Fasta,
    context_sizes: list[int],
    category1: str,
    category2: str,
):
    cat1_variants, cat2_variants = get_eqtls_for_comparison(eqtl_df, category1, category2)
    for context_size in context_sizes:
        cat1_sequences, cat1_intervals = get_sequences(cat1_variants, context_size, fasta)
        cat2_sequences, cat2_intervals = get_sequences(cat2_variants, context_size, fasta)
        
        subdir = f"{category1}_vs_{category2}_{context_size}"
        os.makedirs(subdir, exist_ok=True)
        cat1_fasta_path = os.path.join(subdir, f"{category1}.fasta")
        cat2_fasta_path = os.path.join(subdir, f"{category2}.fasta")
        publish_fasta(cat1_fasta_path, cat1_sequences, cat1_variants, cat1_intervals)
        publish_fasta(cat2_fasta_path, cat2_sequences, cat2_variants, cat2_intervals)


def main():
    args = parse_args()
    eqtl_df = read_eqtl_df(args.eqtl_tsv_path)
    fasta = Fasta(args.fasta_path)

    # Compare consistently_correct vs consistently_correct
    generate_fastas_for_comparison(
        eqtl_df, fasta, args.context_sizes, "consistently_correct", "consistently_incorrect",
    )

    # Compare consistently_correct vs inconsistent
    generate_fastas_for_comparison(
        eqtl_df, fasta, args.context_sizes, "consistently_correct", "inconsistent",
    )


if __name__ == "__main__":
    main()