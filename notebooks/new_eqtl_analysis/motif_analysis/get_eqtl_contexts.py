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
    parser.add_argument("--eqtl_tsv_path", type=str, default="../merged_eqtl_results.tsv")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg38/hg38.fa")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[21, 51, 101, 201, 501, 101])
    return parser.parse_args()


def publish_fasta(output_path: str, sequences: list, variants: list):
    seq_records = []
    for (seq, variant) in zip(sequences, variants):
        seq_records.append(SeqIO.SeqRecord(seq=Seq.Seq(seq), id=variant, description=""))
    SeqIO.write(seq_records, output_path, "fasta")


def main():
    args = parse_args()

    eqtl_df = pd.read_csv(args.eqtl_tsv_path, sep="\t", header=0, index_col=0)
    eqtl_df = eqtl_df[eqtl_df["finemapped"]].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    fasta = Fasta(args.fasta_path)
    for context_size in tqdm(args.context_sizes):
        sequences = defaultdict(list)
        variants = defaultdict(list)

        for (variant, n_correct) in zip(eqtl_df["variant"], eqtl_df["n_correct_CAGE_SAD"]):
            chrom, variant_pos, ref, *_ = variant.split("_")
            variant_pos = int(variant_pos)

            start_pos = variant_pos - context_size // 2
            end_pos = start_pos + context_size - 1
            assert start_pos >= 0
            seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()   
            assert len(seq) == context_size         
            assert seq[context_size // 2] == ref.upper()

            if n_correct == 5:
                category = "consistently_correct"
            elif n_correct == 0:
                category = "consistently_incorrect"
            else:
                category = "inconsistent"

            sequences[category].append(seq)
            variants[category].append(variant)
        
        publish_fasta(
            os.path.join(args.output_dir, f"consistently_correct_{context_size}.fasta"),
            sequences["consistently_correct"],
            variants["consistently_correct"]
        )
        publish_fasta(
            os.path.join(args.output_dir, f"consistently_incorrect_{context_size}.fasta"),
            sequences["consistently_incorrect"],
            variants["consistently_incorrect"]
        )
        publish_fasta(
            os.path.join(args.output_dir, f"inconsistent_{context_size}.fasta"),
            sequences["inconsistent"],
            variants["inconsistent"]
        )




if __name__ == "__main__":
    main()