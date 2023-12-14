import json
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd
import tensorflow as tf
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.append("/data/yosef3/scratch/ruchir/repos/basenji")
from basenji import dna_io
from basenji import seqnn


DNASE_TRACK_IDXS = {
    "Pancreas": 257, "Ovary": 439, "Liver": 452, "Uterus": 283, 
    "Testis": 665, "Spleen": 594, "Lung": 245, "Thyroid": 241, 
    "Prostate": 653, "Vagina": 382, "Stomach": 204, "Adrenal_Gland": 265,
    "Cells_EBV-transformed_lymphocytes": 69
}


CAGE_TRACK_IDXS = {
    "Pancreas": 4946, "Ovary": 4688, "Liver": 4686, "Uterus": 4910,
    "Testis": 4694, "Spleen": 4693, "Lung": 4687, "Thyroid": 4696,
    "Prostate": 4690, "Vagina": 5175, "Stomach": 4959, "Adrenal_Gland": 4977, 
    "Cells_EBV-transformed_lymphocytes": 5110
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("output_npz", type=str)
    parser.add_argument("--eqtl_tsv_path", type=str, default="../merged_eqtl_results.tsv")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg38/hg38.fa")
    parser.add_argument("--params_path", type=str, default="/data/yosef3/scratch/ruchir/repos/basenji/manuscripts/cross2020/params_human.json")
    parser.add_argument("--use_cage", type=bool, action=BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    assert len(gpus) == 1
    tf.config.experimental.set_memory_growth(gpus[0], True)


def prepare_model(model_params: dict, model_path: str):
    seqnn_model = seqnn.SeqNN(model_params)
    seqnn_model.restore(model_path)
    seqnn_model.build_ensemble(ensemble_rc=True, ensemble_shifts=[-1, 0, 1])
    return seqnn_model


def get_eqtls(eqtl_tsv_path) -> tuple[list, list, list]:
    eqtl_df = pd.read_csv(eqtl_tsv_path, sep="\t", header=0, index_col=0)
    eqtl_df = eqtl_df[eqtl_df["finemapped"]].copy()
    variant_idxs = eqtl_df.index.tolist()
    variants = eqtl_df["variant"].tolist()
    tissues = eqtl_df["tissue"].tolist()
    return variant_idxs, variants, tissues


def load_sequence(variant: str, fasta: Fasta, seq_length: int) -> str:
    chrom, variant_pos, ref, *_ = variant.split("_")
    variant_pos = int(variant_pos)

    start_pos = variant_pos - seq_length // 2
    end_pos = start_pos + seq_length - 1

    if start_pos < 0:
        seq = fasta[chrom][0: end_pos].seq.upper()
        seq = "N" * (seq_length - len(seq)) + seq
    else:
        seq = fasta[chrom][start_pos - 1: end_pos].seq.upper()
        if len(seq) < seq_length:
            seq = seq + "N" * (seq_length - len(seq))
    
    variant_idx = variant_pos - start_pos
    assert seq[variant_idx] == ref.upper()
    return seq
    

# Copied from /data/yosef3/users/ruchir/pgp_uq/gradients/get_gradients.py
def get_seq_gradient(seqnn_model, seq: str, track_idx: int) -> np.ndarray:
    seq_1hot = dna_io.dna_1hot(seq)
    seq_1hot = tf.convert_to_tensor(seq_1hot, dtype=tf.float32)
    assert len(seq_1hot.shape) == 2
    batched_seq_1hot = tf.expand_dims(seq_1hot, axis=0)

    input_sequence = tf.keras.Input(shape=seq_1hot.shape, name="sequence")
    full_preds = seqnn_model.ensemble(input_sequence) # (batch_size, num_bins, num_tracks)
    target_slice = tf.gather(full_preds, [track_idx], axis=2)
    center_slice = tf.gather(
        target_slice,
        np.arange(target_slice.shape[1] // 2 - 1, target_slice.shape[1] // 2 + 2),
        axis=1
    )
    model_batch = tf.keras.Model(inputs=input_sequence, outputs=center_slice)

    with tf.GradientTape() as tape:
        tape.watch(batched_seq_1hot)
        output = model_batch(batched_seq_1hot, training=False)
        output = tf.reduce_mean(output, axis=1)

        grads = tape.gradient(output, batched_seq_1hot)
        grads = tf.squeeze(grads)
    return grads.numpy().astype(np.float16)


def main():
    args = parse_args()
    tissue_to_track_idxs = CAGE_TRACK_IDXS if args.use_cage else DNASE_TRACK_IDXS
    print(f"Using {tissue_to_track_idxs} for track idxs")

    configure_gpus()
    model_params = json.load(open(args.params_path))["model"]
    seqnn_model = prepare_model(model_params, args.model_path)

    all_grads = []
    variant_idxs, variants, tissues = get_eqtls(args.eqtl_tsv_path)
    track_idxs = [tissue_to_track_idxs[t] for t in tissues]
    
    fasta = Fasta(args.fasta_path) 
    for variant, track_idx in tqdm(zip(variants, track_idxs)):
        seq = load_sequence(variant, fasta, model_params["seq_length"])
        grads = get_seq_gradient(seqnn_model, seq, track_idx)
        all_grads.append(grads)
    
    all_grads = np.asarray(all_grads)
    np.savez(
        args.output_npz,
        variant_idxs=variant_idxs,
        variants=variants,
        tissues=tissues,
        track_idxs=track_idxs,
        grads=all_grads
    )


if __name__ == "__main__":
    main()


