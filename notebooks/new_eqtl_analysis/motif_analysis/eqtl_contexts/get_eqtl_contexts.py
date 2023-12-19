import os
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from Bio import Seq, SeqIO
from pyfaidx import Fasta
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--eqtl_tsv_path", type=str, default="../../merged_eqtl_results.tsv")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg38/hg38.fa")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[21, 51, 101, 201, 501, 1001])
    return parser.parse_args()


def publish_fasta(output_path: str, sequences: list, variants: list, intervals: list):
    assert len(sequences) == len(variants) == len(intervals)
    assert len(set(intervals)) == len(intervals)

    seq_records = [
        SeqIO.SeqRecord(Seq.Seq(seq), id=interval, description=variant) 
        for (seq, interval, variant) in zip(sequences, intervals, variants)
    ]
    SeqIO.write(seq_records, output_path, "fasta")


def get_eqtls(eqtl_df: pd.DataFrame, context_size: int, fasta: Fasta) -> tuple[list, list, list]:
    """
    Returns three dicts: variants, sequences, and intervals. For each dictionary, the key is a 
    category in ["consistent", "consistently_correct", "consistently_incorrect", "inconsistent"]/
    """
    variants, sequences, intervals = defaultdict(list), defaultdict(list), defaultdict(list)

    category_map = {
        5: ["consistent", "consistently_correct"],
        0: ["consistent", "consistently_incorrect"],
    }

    for (variant, n_correct) in zip(eqtl_df["variant"], eqtl_df["n_correct_CAGE_SAD"]):
        chrom, variant_pos, ref, *_ = variant.split("_")
        variant_pos = int(variant_pos)
        start_pos = variant_pos - context_size // 2
        end_pos = start_pos + context_size - 1
        seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()
        interval = f"{chrom}:{start_pos}-{end_pos}"

        assert len(seq) == context_size
        assert seq[context_size // 2] == ref.upper()

        categories = category_map.get(n_correct, ["inconsistent"])
        for category in categories:
            variants[category].append(variant)
            sequences[category].append(seq)
            intervals[category].append(interval)
        
    return variants, sequences, intervals


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eqtl_df = pd.read_csv(args.eqtl_tsv_path, sep="\t", header=0, index_col=0)
    eqtl_df = eqtl_df[eqtl_df["finemapped"]].copy()

    fasta = Fasta(args.fasta_path)

    for tissue, tissue_df in tqdm(eqtl_df.groupby("tissue")):
        for context_size in tqdm(args.context_sizes):
            variants, sequences, intervals = get_eqtls(tissue_df, context_size, fasta)
            subdir = os.path.join(args.output_dir, f"{tissue}_{context_size}")
            os.makedirs(subdir, exist_ok=True)
            for category in [
                "consistent", "consistently_correct", "consistently_incorrect", "inconsistent"
            ]:
                output_path = os.path.join(subdir, f"{category}.fasta")
                publish_fasta(output_path, sequences[category], variants[category], intervals[category])


if __name__ == "__main__":
    main()