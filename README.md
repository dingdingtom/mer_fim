## FIM

### Title

Focuses on Informative Modalities: Towards Modality Consistent Multi-modal Emotion Recognition

### Requirements

CUDA 12.2

```bash
pip install -r requirements.txt
```

### Data

[CHERMA-data](https://github.com/sunjunaimer/LFMIM)

[CHERMA-paper](https://aclanthology.org/2023.acl-long.39v2.pdf)

### Structure

```
|__ checkpoint 
|__ data
|__ modules
|   |__ multihead_attention.py
|   |__ position_embedding.py
|   |__ transformer.py
|__ data_prepare.py
|__ model.py
|__ trainer.py
|__ run_train.sh
|__ run_test.sh
|__ requirements.txt
```

### Results

CHERMA

```bash
bash run_cherma_train.sh
bash run_cherma_test.sh
```

Note. Please rewrite `dataset_dir` during training, and rewrite `dataset_dir`, `path_ckpt` during testing. You can follow the Structure section above to save checkpoints and data.

### Acknowledgement

The code is modified upon the released code of 2023 ACL paper " Layer-wise Fusion with Modality Independence Modeling for Multi-modal Emotion Recognition ".

We appreciate their open-sourcing such high-quality code and dataset, which is very helpful to our research. We thank pytorch and pytorch-lightning for their wonderful training implementation. 

