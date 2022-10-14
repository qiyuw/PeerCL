# PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings
**Update: PCL has been accepted to the main conference of EMNLP 2022.**

This repository includes the source codes of paper [PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings](https://arxiv.org/abs/2201.12093).
Part of the implementation of Demo, baselines and evaluation are from [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Get started
| Model List|
|-------|
|[qiyuw/pcl-bert-base-uncased](https://huggingface.co/qiyuw/pcl-bert-base-uncased) |
|[qiyuw/pcl-roberta-base](https://huggingface.co/qiyuw/pcl-roberta-base) |
|[qiyuw/pcl-bert-large-uncased](https://huggingface.co/qiyuw/pcl-bert-large-uncased) |
|[qiyuw/pcl-roberta-large](https://huggingface.co/qiyuw/pcl-roberta-large) |

Use the pre-trained model with [huggingface](https://huggingface.co/)

```
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("qiyuw/pcl-bert-base-uncased")
model = AutoModel.from_pretrained("qiyuw/pcl-bert-base-uncased")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```

## Demo
Run the simple demo of information retrieval by `python pcl/tool.py --model_name_or_path MODEL_NAME`. `MODEL_NAME` here can be any name or path of the well-trained model.

## Preparing data
Get training data by running `bash download_wiki.sh`

Get evaluation data by running `bash PCL/SentEval/data/downstream/download_dataset.sh`

## Train
Currently please train the model on single GPU.

Train PCL by running
```
mkdir result

python train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/wiki1m_for_simcse.txt \
  --output_dir result \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --learning_rate 3e-5 \
  --max_seq_length 32 \
  --evaluation_strategy steps \
  --metric_for_best_model stsb_spearman \
  --load_best_model_at_end \
  --eval_steps 125 \
  --pooler_type cls \
  --mlp_only_train \
  --overwrite_output_dir \
  --temp 0.05 \
  --do_train \
  --do_eval \
  --fp16 \
  --no_extend_neg_samples \
  "$@"
```

## Evaluation
Evaluate the model by `python evaluation.py --model_name_or_path MODEL_NAME --mode test --pooler cls_before_pooler`. `MODEL_NAME` here can be any name or path of the well-trained model.

The results of unsupervised PCL on STS bencemarks are as follows:
Model        | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
-------------|-------|-------|-------|-------|-------|--------------|-----------------|-------|
bert-base    | 73.87 | 82.60 | 75.71 | 82.67 | 80.22 |    79.55     |      72.12      | 78.11 |
roberta-base | 70.54 | 83.25 | 75.73 | 83.46 | 81.81 |    81.83     |      69.27      | 77.98 |
bert-large   | 74.92 | 86.01 | 78.92 | 85.11 | 80.06 |    81.33     |      73.53      | 79.98 |
roberta-large| 72.76 | 84.72 | 77.49 | 85.03 | 81.78 |    83.26     |      73.49      | 79.79 |

The results of unsupervised PCL on transfer tasks are as follows:
Model        |   MR  |   CR  |  SUBJ |  MPQA |  SST2 |  TREC |  MRPC |  Avg. |
-------------|-------|-------|-------|-------|-------|-------|-------|-------|
bert-base    | 80.11 | 85.27 | 94.22 | 89.15 | 85.50 | 87.40 | 76.12 | 85.40 |
roberta-base | 81.86 | 87.55 | 92.98 | 87.20 | 87.26 | 85.20 | 76.46 | 85.50 |
bert-large   | 82.45 | 87.84 | 95.04 | 89.61 | 87.81 | 93.00 | 75.94 | 87.38 |
roberta-large| 84.49 | 89.06 | 94.67 | 89.26 | 89.07 | 94.20 | 74.90 | 87.95 |

## Citation
Cite our paper if PCL helps your work:

```bibtex
@article{wu2022pcl,
  title={PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings},
  author={Wu, Qiyu and Tao, Chongyang and Shen, Tao and Xu, Can and Geng, Xiubo and Jiang, Daxin},
  journal={arXiv preprint arXiv:2201.12093},
  year={2022}
}
```
