# PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings
This repository includes the source codes of paper [PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings](https://arxiv.org/abs/2201.12093).
The implementation of Demo, baselines and evaluation are from [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Demo
Run the simple demo of information retrieval by `python pcl/tool.py --model_name_or_path MODEL_NAME`. `MODEL_NAME` here can be any name or path of the well-trained model.

## Preparing data
TODO

## Train
Currently please train the model on single GPU.

## Evaluation
Evaluate the model by `python evaluation.py --model_name_or_path MODEL_NAME --mode test --pooler cls_before_pooler`. `MODEL_NAME` here can be any name or path of the well-trained model.

The results of unsupervised PCL on STS bencemarks are as follows:
| bert-base | roberta-base | bert-large | roberta-large |
|-----------|--------------|------------|---------------|
| 78.11     | 77.98        | 79.98      | 79.79         |

Model        | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
-------------|-------|-------|-------|-------|-------|--------------|-----------------|-------|
bert-base    | 73.87 | 82.60 | 75.71 | 82.67 | 80.22 |    79.55     |      72.12      | 78.11 |
roberta-base | 70.54 | 83.25 | 75.73 | 83.46 | 81.81 |    81.83     |      69.27      | 77.98 |
bert-large   | | | | | | | |79.98|
roberta-large| | | | | | | |79.79|

The results of unsupervised PCL on transfer tasks are as follows:
Model        |   MR  |   CR  |  SUBJ |  MPQA |  SST2 |  TREC |  MRPC |  Avg. |
-------------|-------|-------|-------|-------|-------|-------|-------|-------|
bert-base    | 80.11 | 85.27 | 94.22 | 89.15 | 85.50 | 87.40 | 76.12 | 85.40 |
roberta-base | 81.86 | 87.55 | 92.98 | 87.20 | 87.26 | 85.20 | 76.46 | 85.50 |
bert-large   | | | | | | | | |
roberta-large| | | | | | | | |

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
