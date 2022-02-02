# KGML


## About
Source code of the paper [Knowledge-Aware Meta-learning for Low-Resource Text Classification](https://arxiv.org/abs/2109.04707).

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2021knowledge,
  title={Knowledge-Aware Meta-learning for Low-Resource Text Classification},
  author={Yao, Huaxiu and Wu, Yingxin and Al-Shedivat, Maruan and Wei, Ying and Xing, Eric P.},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021} 
}
```

## Preprocess Knowledge Graph

1. Download LibKGE in folder `kg_preprocess/`

     ```git clone git@github.com:uma-pi1/kge.git```

2. Pretrain Knowledge Graph

    ```python -m kge start rescal.yaml```

3. Preprocess Knowledge Graph, c.f. `knowledge_subgraph.ipynb` (Note the name of model file needs to be change accrodingly).

After preprocessing, the following files will be in the folder `data/wn18rr_kg`

```
vocab.pt  
wn18rr_edge_attr.pt  
wn18rr_edge_index.pt  
wn18rr_edge_type.pt  
wn18rr_knn_edge_index.pt  
wn18rr_x.pt
```

## Unsupervised KGML
### I. Datasets

- **Twitter**

    Please refer to https://github.com/TalwalkarLab/leaf/tree/master/data/sent140 and run `preprocess.sh` via
    ```
    ./preprocess.sh -s niid --sf 1.0 -k 50 -t user --tf 0.8
    ```
    Put generated data and embedding file into folder `data/twitter`. Ensure there are four file/folders under `data/twitter`:
    ```
    test/     train/   val/   embs.json
 
### II. Training

```
SEED=0
python train.py --dataset twitter  --use_context 0  --use_kg 1 --experiment_name twitter_kg_$SEED --meta_batch_size 2 --support_size 50 --sampling_type meta_batch_groups --uniform_over_groups 10  --num_epochs 200 --epochs_per_eval 1 --n_test_per_dist 2000 --optimizer adam --seed $SEED 
```
### III. Testing
```
python test_on_groups.py --dataset twitter  --eval_on test --use_context 0 --use_kg 1 --ckpt_folders twitter_kg_0_0_xxxx --n_test_dists 30
```

## Supervised KGML

### I. Datasets

We put preprocessed data in [Google Drive](https://drive.google.com/drive/folders/16CZAi9_FgiulK7m7bXrnldkDMzjjnRzA?usp=sharing)

### II. Training and Testing

See run.sh in supervised folder for more details

