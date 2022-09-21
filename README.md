# PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings
This repository includes the source codes of paper [PCL: Peer-Contrastive Learning with Diverse Augmentations for Unsupervised Sentence Embeddings](https://arxiv.org/abs/2201.12093).
The implementation of Demo, baselines and evaluation are from [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Demo
Run the simple demo of information retrieval by `python pcl/tool.py --model_name bert-base-uncased`. `bert-base-uncased` here can be any name or path of the well-trained model.

## Preparing data
TODO

## Train
Currently please train the model on single GPU.

## Evaluation
TODO

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
