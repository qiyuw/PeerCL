# PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings
This repository includes the source codes of paper [PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings](https://arxiv.org/abs/2201.12093).
The implementation of Demo, baselines and evaluation are from [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Demo
Run the simple demo of information retrieval by `python pcl/tool.py --model_name MODEL_NAME`. `MODEL_NAME` here can be any name or path of the well-trained model.

## Preparing data
TODO

## Train
Currently please train the model on single GPU.

## Evaluation
Evaluate the model by `python evaluation.py --model_name_or_path MODEL_NAME --mode test --pooler cls_before_pooler`. `MODEL_NAME` here can be any name or path of the well-trained model.

The average results of unsupervised PCL on STS bencemarks are as follows:
| bert-base | roberta-base | bert-large | roberta-large |
|-----------|--------------|------------|---------------|
| 78.11     | 77.98        | 79.98      | 79.79         |

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
