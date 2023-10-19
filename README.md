# basenji2_uncertainty

This repository contains information and scripts to characterize uncertainty in predictions of genomic sequence-to-activity models. It makes use of scripts from the [Basenji](https://github.com/calico/basenji/tree/master) repository for training a deep ensemble of Basenji2 models, generating, and evaluating predictions.

## Training ensemble of Basenji2 models

Command used to train each model replicate:
```
basenji_train.py -k -o ${out_dir}/train/rep_${replicate_model}/ ${out_dir}/models/params_human.json ${data_dir}/human
```

Necessary data and resources:
* Basenji2 training, validation and test data can be downloaded from Google Cloud ([link](https://console.cloud.google.com/storage/browser/basenji_barnyard/data)). Note: This data is ~320 GB and is in a requester pays bucket.
* `params_human.json` can be found [here](https://github.com/calico/basenji/blob/master/manuscripts/cross2020/params_human.json)

## Generating predictions on held-out reference sequences

Command used to generate test set predictions for each model replicate:
```
basenji_test.py --save --rc --shifts "1,0,-1" -t ${data_dir}/human/targets.txt \
                -o ${out_dir}/test/rep_${replicate_model}/ \
                ${out_dir}/models/params_human.json ${out_dir}/train/rep_${replicate_model}/model_best.h5 ${data_dir}/human
```
The same data and resources are necessary as the training step above.

## Generating predictions for fine-mapped GTeX eQTLs

Command used to generate predictions for each model replicate and each tissue:
```
for eqtl_set in pos neg;
do
  basenji_sad.py --rc --shifts -1,0,1 --stats SAD,REF,ALT \
                 -o ${out_dir}/preds/basenji2_${replicate_model}_${tissue}_${eqtl_set} \
                 -t ${data_dir}/human/targets.txt -f ${hg38_fasta} \
                 ${out_dir}/models/params_human.json \
                 ${out_dir}/train/rep_${replicate_model}/model_best.h5 \
                 ${gtex_dir}/vcf/${tissue}_${eqtl_set}.vcf
done
```
Necessary data and resources:
* GTeX SuSie fine-mapped eQTL data from [Wang et al. (2021)](https://www.nature.com/articles/s41592-021-01252-x#ref-CR22) and [Avsec et al. (2021)](https://www.nature.com/articles/s41592-021-01252-x) can be downloaded from Google Cloud ([link](https://console.cloud.google.com/storage/browser/dm-enformer/data/gtex_fine))
