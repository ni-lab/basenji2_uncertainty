REPLICATES_MODEL_DIR="/data/yosef3/users/ruchir/pgp_uq/data/reps/"

for i in {1..5}
do
    python get_gradients.py --use_cage ${REPLICATES_MODEL_DIR}/rep${i}.h5 cage_rep${i}.npz
done