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
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[21, 51, 101, 201, 501, 1001])
    return parser.parse_args()


def publish_fasta(output_path: str, sequences: list, variants: list, tissues: list):
    assert len(sequences) == len(variants) == len(tissues)
    
    ids = [f"{variant}:{tissue}" for (variant, tissue) in zip(variants, tissues)]
    assert len(set(ids)) == len(ids)

    seq_records = [
        SeqIO.SeqRecord(Seq.Seq(seq), id=id_, description="") for (seq, id_) in zip(sequences, ids)
    ]
    SeqIO.write(seq_records, output_path, "fasta")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eqtl_df = pd.read_csv(args.eqtl_tsv_path, sep="\t", header=0, index_col=0)
    eqtl_df = eqtl_df[eqtl_df["finemapped"]].copy()

    fasta = Fasta(args.fasta_path)
    category_map = {
        5: ["consistent", "consistently_correct"],
        0: ["consistent", "consistently_incorrect"],
    }

    for context_size in tqdm(args.context_sizes):
        sequences, variants, tissues = defaultdict(list), defaultdict(list), defaultdict(list)
    
        for (variant, tissue, n_correct) in zip(
            eqtl_df["variant"], eqtl_df["tissue"], eqtl_df["n_correct_CAGE_SAD"]
        ):
            chrom, variant_pos, ref, *_ = variant.split("_")
            variant_pos = int(variant_pos)
            start_pos = variant_pos - context_size // 2
            end_pos = start_pos + context_size - 1
            seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()   
            assert len(seq) == context_size         
            assert seq[context_size // 2] == ref.upper()

            categories = category_map.get(n_correct, ["inconsistent"])
            for category in categories:
                sequences[category].append(seq)
                variants[category].append(variant)
                tissues[category].append(tissue)

        for category in ["consistent", "consistently_correct", "consistently_incorrect", "inconsistent"]:
            output_path = os.path.join(args.output_dir, f"{category}_{context_size}.fasta")
            publish_fasta(output_path, sequences[category], variants[category], tissues[category])


if __name__ == "__main__":
    main()