import numpy as np
import pandas as pd
from Bio import Seq, SeqIO
from pyfaidx import Fasta

DRIVERS_TSV_PATH = "/data/yosef3/scratch/ruchir/repos/EnformerAssessment/Data/SupplementaryTable2.tsv"
FASTA_PATH = "/data/yosef3/scratch/ruchir/data/genomes/hg38/hg38.fa"
CONTEXT_SIZE = 51
SUPPORTED_OUTPATH = f"supported_{CONTEXT_SIZE}.fasta"
UNSUPPORTED_OUTPATH = f"unsupported_{CONTEXT_SIZE}.fasta"


def read_drivers_df():
    drivers_df = pd.read_csv(DRIVERS_TSV_PATH, sep="\t")
    drivers_df["chr_hg38"] = "chr" + drivers_df["chr_hg38"].astype(str)
    drivers_df["variant_hg38"] = drivers_df["chr_hg38"] + ":" + drivers_df["LociDriverhg38"].astype(str)
    drivers_df["supported"] = [
        np.sign(ism) == np.sign(beta) 
        for ism, beta in zip(drivers_df["ISM"], drivers_df["eQTL"])
    ]
    return drivers_df


def get_sequences(variants, fasta):
    sequences = []
    for v in variants:
        chrom, pos = v.split(":")
        start_pos = int(pos) - CONTEXT_SIZE // 2
        end_pos = start_pos + CONTEXT_SIZE - 1
        seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()
        assert len(seq) == CONTEXT_SIZE
        sequences.append(seq)
    return sequences


def get_gc_content(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def publish_fasta(output_path: str, sequences: list, variants: list):
    assert len(sequences) == len(variants)
    seq_records = [
        SeqIO.SeqRecord(Seq.Seq(seq), id=variant, description="")
        for (seq, variant) in zip(sequences, variants)
    ]
    SeqIO.write(seq_records, output_path, "fasta")


def main():
    drivers_df = read_drivers_df()
    supported_variants = list(set(drivers_df[drivers_df["supported"]]["variant_hg38"].values))
    unsupported_variants = list(set(drivers_df[~drivers_df["supported"]]["variant_hg38"].values))

    fasta = Fasta(FASTA_PATH)
    supported_seqs = get_sequences(supported_variants, fasta)
    unsupported_seqs = get_sequences(unsupported_variants, fasta)
    supported_seq_gcs = [get_gc_content(seq) for seq in supported_seqs]
    unsupported_seq_gcs = [get_gc_content(seq) for seq in unsupported_seqs]
    print(f"# supported: {len(supported_seqs)}, avg GC: {np.mean(supported_seq_gcs):.3f}")
    print(f"# unsupported: {len(unsupported_seqs)}, avg GC: {np.mean(unsupported_seq_gcs):.3f}")
    
    publish_fasta(SUPPORTED_OUTPATH, supported_seqs, supported_variants)
    publish_fasta(UNSUPPORTED_OUTPATH, unsupported_seqs, unsupported_variants)


if __name__ == "__main__":
    main()